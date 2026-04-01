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
import com.google.gson.JsonObject
import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi

/** Represents a message in the conversation. A message can contain multiple [Content]. */
class Message private constructor(val contents: List<Content>) {

  fun init() {
    check(contents.isNotEmpty()) { "Contents should not be empty." }
  }

  /** Convert to [JsonArray]. Used internally. */
  internal fun toJson(): JsonArray {
    return JsonArray().apply {
      for (content in contents) {
        this.add(content.toJson())
      }
    }
  }

  override fun toString(): String {
    return contents.joinToString("")
  }

  companion object {
    /** Creates a [Message] from a text string. */
    fun of(text: String): Message {
      return Message(listOf(Content.Text(text)))
    }

    /** Creates a [Message] from the array of [Content]. */
    fun of(vararg contents: Content): Message {
      return Message(contents.toList())
    }

    /** Creates a [Message] from a list of [Content]. */
    fun of(contents: List<Content>): Message {
      return Message(contents)
    }
  }
}

/** Represents a content in the [Message] of the conversation. */
sealed class Content {
  /** Convert to [JsonObject]. Used internally. */
  internal abstract fun toJson(): JsonObject

  /** Text. */
  data class Text(val text: String) : Content() {
    override fun toJson(): JsonObject {
      return JsonObject().apply {
        addProperty("type", "text")
        addProperty("text", text)
      }
    }

    override fun toString(): String {
      return text
    }
  }

  /** Image provided as raw bytes. */
  @OptIn(ExperimentalEncodingApi::class)
  data class ImageBytes(val bytes: ByteArray) : Content() {
    override fun toJson(): JsonObject {
      return JsonObject().apply {
        addProperty("type", "image")
        addProperty("blob", Base64.encode(bytes))
      }
    }
  }

  /** Image provided by a file. */
  data class ImageFile(val absolutePath: String) : Content() {
    override fun toJson(): JsonObject {
      return JsonObject().apply {
        addProperty("type", "image")
        addProperty("path", absolutePath)
      }
    }
  }

  /** Audio provided as raw bytes. */
  @OptIn(ExperimentalEncodingApi::class)
  data class AudioBytes(val bytes: ByteArray) : Content() {
    override fun toJson(): JsonObject {
      return JsonObject().apply {
        addProperty("type", "audio")
        addProperty("blob", Base64.encode(bytes))
      }
    }
  }

  /** Audio provided by a file. */
  data class AudioFile(val absolutePath: String) : Content() {
    override fun toJson(): JsonObject {
      return JsonObject().apply {
        addProperty("type", "audio")
        addProperty("path", absolutePath)
      }
    }
  }
}
