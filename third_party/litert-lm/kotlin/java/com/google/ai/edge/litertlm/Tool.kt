/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ai.edge.litertlm

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive
import kotlin.reflect.KParameter
import kotlin.reflect.full.functions

/**
 * Example of how to define tools:
 * - Use `@Tool` to define a method as a tool.
 * - Use `@ToolParam` to add information to the param of a tool.
 * - The allowed parameter types are: String, Int, Boolean, Float, Double, and List of them..
 * - The return type could be anything and will to converted with toString() back to the model.
 * - Use the Kotlin nullable type (e.g., String?) to indicate that a parameter is optional.
 *
 * ```kotlin
 * class MyToolSet {
 *   @Tool(description = "Get the current weather")
 *   fun getCurrentWeather(
 *     @ToolParam(description = "The city and state, e.g. San Francisco, CA") location: String,
 *     @ToolParam(description = "The temperature unit to use") unit: String = "celsius",
 *   ): Map {
 *     return mapOf(
 *       "temperature" to 25,
 *       "unit" to "Celsius",
 *     )
 *   }
 * }
 * ```
 */

/**
 * Annotation to mark a function as a tool that can be used by the LiteRT-LM model.
 *
 * @property description A description of the tool.
 */
@Target(AnnotationTarget.FUNCTION) // This annotation can only be applied to functions
@Retention(AnnotationRetention.RUNTIME) // IMPORTANT: Makes the annotation available at runtime
annotation class Tool(val description: String)

/**
 * Annotation to provide a description for a tool parameter.
 *
 * @property description A description of the tool parameter.
 */
@Target(AnnotationTarget.VALUE_PARAMETER) // This annotation can only be applied to functions
@Retention(AnnotationRetention.RUNTIME) // IMPORTANT: Makes the annotation available at runtime
annotation class ToolParam(val description: String)

/**
 * Represents a single tool, wrapping an instance and a specific Kotlin function.
 *
 * @property instance The instance of the class containing the tool function.
 * @property kFunction The Kotlin function to be executed as a tool.
 * @property useSnakeCase Whether to use snake case for function and param names for tool calling.
 */
internal class Tooling(
  val instance: Any,
  val kFunction: kotlin.reflect.KFunction<*>,
  val useSnakeCase: Boolean,
) {

  companion object {
    private val javaTypeToJsonTypeString =
      mapOf(
        String::class to "string",
        Int::class to "integer",
        Boolean::class to "boolean",
        Float::class to "number",
        Double::class to "number",
        List::class to "array",
      )
  }

  private fun KParameter.toModelParamName(): String {
    return if (useSnakeCase) this.name!!.camelToSnakeCase() else this.name!!
  }

  /**
   * Executes the tool function with the given parameters.
   *
   * @param params A JsonObject containing the parameter names and their values.
   * @return The result of the tool function execution as a Any?.
   * @throws IllegalArgumentException if any required parameters are missing.
   */
  fun execute(params: JsonObject): Any? {
    val args =
      kFunction.parameters
        .associateWith { param ->
          when {
            param.index == 0 -> instance // First parameter is the instance
            param.name != null && params.has(param.toModelParamName()) -> {
              val value = params.get(param.toModelParamName())
              convertJsonValueToKotlinValue(value, param.type)
            }
            param.isOptional -> null // Should not be reached
            else -> throw IllegalArgumentException("Missing parameter: ${param.toModelParamName()}")
          }
        }
        .filterValues { it != null }

    return kFunction.callBy(args)
  }

  /**
   * Converts a JSON value to the expected Kotlin type.
   *
   * @param value The JSON value to convert.
   * @param type The target Kotlin type.
   * @return The converted value.
   * @throws IllegalArgumentException if the value cannot be converted to the target type.
   */
  private fun convertJsonValueToKotlinValue(value: JsonElement, type: kotlin.reflect.KType): Any {
    val classifier = type.classifier
    return when {
      classifier == List::class && value is JsonArray -> {
        val listTypeArgument = type.arguments.firstOrNull()?.type
        value.map { convertJsonValueToKotlinValue(it, listTypeArgument!!) }
      }
      classifier == Int::class && value is JsonPrimitive && value.isNumber -> value.asInt
      classifier == Float::class && value is JsonPrimitive && value.isNumber -> value.asFloat
      classifier == Double::class && value is JsonPrimitive && value.isNumber -> value.asDouble
      classifier == String::class && value is JsonPrimitive && value.isString -> value.asString
      classifier == Boolean::class && value is JsonPrimitive && value.isBoolean -> value.asBoolean
      // Add more conversions if needed
      else -> value
    }
  }

  /**
   * Generates a JSON schema for the given Kotlin type.
   *
   * @param type The Kotlin type to generate the schema for.
   * @return A JsonObject representing the JSON schema.
   * @throws IllegalArgumentException if the type is not supported.
   */
  private fun getTypeJsonSchema(type: kotlin.reflect.KType): JsonObject {
    val classifier = type.classifier
    val jsonType = javaTypeToJsonTypeString[classifier]

    if (jsonType == null) {
      throw IllegalArgumentException(
        "Unsupported type: ${classifier.toString()}. " +
          "Allowed types are: ${javaTypeToJsonTypeString.keys.joinToString { it.simpleName ?: "" }}"
      )
    }

    val schema = JsonObject()
    schema.addProperty("type", jsonType)
    if (classifier == List::class) {
      val listTypeArgument = type.arguments.firstOrNull()?.type
      if (listTypeArgument == null) {
        throw IllegalArgumentException("List type argument is missing.")
      }
      schema.add("items", getTypeJsonSchema(listTypeArgument))
    }
    return schema
  }

  /**
   * Gets the tool description in Open API format.
   *
   * @return The tool description.
   */
  fun getToolDescription(): JsonObject {
    val toolAnnotation = kFunction.annotations.find { it is Tool } as? Tool ?: return JsonObject()

    val description = toolAnnotation.description

    val openApiSpec =
      JsonObject().apply {
        val funcName = if (useSnakeCase) kFunction.name.camelToSnakeCase() else kFunction.name
        addProperty("name", funcName)
        addProperty("description", description)
      }

    val parameters = kFunction.parameters.drop(1) // Drop the instance parameter
    if (!parameters.isEmpty()) {
      val properties = JsonObject()
      for (param in parameters) {
        val paramAnnotation = param.annotations.find { it is ToolParam } as? ToolParam
        val paramJsonSchema = getTypeJsonSchema(param.type)
        // add "description" if provided
        paramAnnotation?.description?.let { paramJsonSchema.addProperty("description", it) }
        if (param.type.isMarkedNullable) paramJsonSchema.addProperty("nullable", true)
        properties.add(param.toModelParamName(), paramJsonSchema)
      }

      val requiredParams = JsonArray()
      for (param in parameters) {
        if (!param.isOptional) {
          requiredParams.add(param.toModelParamName())
        }
      }

      val schema =
        JsonObject().apply {
          addProperty("type", "object")
          add("properties", properties)
          if (!requiredParams.isEmpty) add("required", requiredParams)
        }

      openApiSpec.add("parameters", schema)
    }

    // Wrap the Open API spec in function object, expected by the native library.
    return JsonObject().apply {
      addProperty("type", "function")
      add("function", openApiSpec)
    }
  }
}

/**
 * Manages a collection of tool sets and provides methods to execute tools and get their
 * specifications.
 *
 * @property toolSets A list of objects, where each object contains methods annotated with @Tool.
 */
class ToolManager(val toolSets: List<Any>) {

  @OptIn(ExperimentalApi::class)
  private val useSnakeCase = ExperimentalFlags.convertCamelToSnakeCaseInToolDescription

  private val tools: Map<String, Tooling> =
    toolSets
      .flatMap { toolSet ->
        val toolClass = toolSet.javaClass.kotlin
        toolClass.functions
          .filter { function -> function.annotations.any { annotation -> annotation is Tool } }
          .map { function ->
            (if (useSnakeCase) function.name.camelToSnakeCase() else function.name) to
              Tooling(toolSet, function, useSnakeCase)
          }
      }
      .toMap()

  /**
   * Executes a tool function by its name with the given parameters.
   *
   * @param functionName The name of the tool function to execute.
   * @param params A JsonObject containing the parameter names and their values.
   * @return The result of the tool function execution as a string.
   * @throws IllegalArgumentException if the tool function is not found.
   */
  fun execute(functionName: String, params: JsonObject): JsonElement {
    try {
      val tool =
        tools[functionName] ?: throw IllegalArgumentException("Tool not found: ${functionName}")
      return convertKotlinValueToJsonValue(tool.execute(params))
    } catch (e: Exception) {
      return JsonPrimitive("Error occured. ${e.toString()}")
    }
  }

  /**
   * Gets the tools description for all registered tools in Open API format.
   *
   * @return A json array of OpenAPI tool description JSON as string.
   */
  fun getToolsDescription(): JsonArray {
    val array = JsonArray()
    for (tool in tools.values) {
      array.add(tool.getToolDescription())
    }
    return array
  }

  private fun convertKotlinValueToJsonValue(kValue: Any?): JsonElement {
    return when (kValue) {
      is List<*> -> {
        val array = JsonArray()
        for (item in kValue) {
          if (item != null) {
            array.add(convertKotlinValueToJsonValue(item))
          }
        }
        array
      }
      is Map<*, *> -> {
        val obj = JsonObject()
        for ((key, value) in kValue) {
          if (key != null && value != null) {
            obj.add(key.toString(), convertKotlinValueToJsonValue(value))
          }
        }
        obj
      }
      is String -> JsonPrimitive(kValue)
      is Number -> JsonPrimitive(kValue)
      is Boolean -> JsonPrimitive(kValue)
      is kotlin.Unit -> JsonPrimitive("") // special case when a Kotlin function return nothing.
      else -> JsonPrimitive(kValue.toString())
    }
  }
}

private fun String.camelToSnakeCase(): String {
  return this.replace(Regex("(?<=[a-zA-Z])[A-Z]")) { "_${it.value}" }.lowercase()
}

private fun String.snakeToCamelCase(): String {
  return Regex("_([a-z])").replace(this) { it.value.substring(1).uppercase() }
}
