# LiteRT-LM Kotlin API

The Kotlin API of LiteRT-LM for **Android** and **JVM (Linux, MacOS, Windows)**
with features like **GPU acceleration** (NPU upcoming!), **multi-modality**, and
**tools use**.

## Introduction

Here is a sample terminal chat app built with the Kotlin API:

```kotlin
import com.google.ai.edge.litertlm.*

suspend fun main() {
  Engine.setNativeMinLogSeverity(LogSeverity.ERROR) // hide log for TUI app

  val engineConfig = EngineConfig(modelPath = "/path/to/model.litertlm")
  Engine(engineConfig).use { engine ->
    engine.initialize()

    engine.createConversation().use { conversation ->
      while (true) {
        print("\n>>> ")
        conversation.sendMessageAsync(Message.of(readln())).collect { print(it) }
      }
    }
  }
}
```

![](./kotlin_demo.gif)

To try out the above sample, clone the repo and run with
[example/Main.kt](./java/com/google/ai/edge/litertlm/example/Main.kt):

```bazel
bazel run -c opt //kotlin/java/com/google/ai/edge/litertlm/example:main -- <abs_model_path>
```

Available `.litertlm` models are on the [HuggingFace LiteRT
Community](https://huggingface.co/litert-community). The above animation was
using the [Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT).

For Android sample, check out the [Google AI Edge
Gallery](https://github.com/google-ai-edge/gallery) app.

## Getting Started with Gradle

While LiteRT-LM is developed with Bazel, we provide the Maven packages for
Gradle/Maven users.

### 1. Add the Gradle dependency

```
dependencies {
    // For Android
    implementation("com.google.ai.edge.litertlm:litertlm-android:latest.release")

    // For JVM (Linux, MacOS, Windows)
    implementation("com.google.ai.edge.litertlm:litertlm-jvm:latest.release")
}
```

You can find the available versions on Google Maven in
[litertlm-android](https://maven.google.com/web/index.html#com.google.ai.edge.litertlm:litertlm-android)
and
[litertlm-jvm](https://maven.google.com/web/index.html#com.google.ai.edge.litertlm:litertlm-jvm).

`latest.release` could be used to get the latest release.

### 2. Initialize the Engine

The `Engine` is the entry point to the API. Initialize it with the model path
and configuration. Remember to close the engine to release resources.

**Note:** The `engine.initialize()` method can take a significant amount of time
(e.g., up to 10 seconds) to load the model. It is strongly recommended to call
this on a background thread or coroutine to avoid blocking the UI thread.

```kotlin
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig

val engineConfig = EngineConfig(
    modelPath = "/path/to/your/model.litertlm", // Replace with your model path
    backend = Backend.CPU, // Or Backend.GPU
    // optional: Pick a writable dir. This can improve 2nd load time.
    // cacheDir = "/tmp/" or context.cacheDir.path (for Android)
)

val engine = Engine(engineConfig)
engine.initialize()
// ... Use the engine to create conversation ...

// Close the engine when done
engine.close()
```

On Android, to use the GPU backend, the app needs to request explicitly by
adding the following to your `AndroidManifest.xml` inside the `<application>`
tag:

```xml
  <application>
    <uses-native-library android:name="libvndksupport.so" android:required="false"/>
    <uses-native-library android:name="libOpenCL.so" android:required="false"/>
  </application>
```

### 3. Create a Conversation

Once the engine is initialized, create a `Conversation` instance. You can
provide a `ConversationConfig` to customize its behavior.

```kotlin
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig

// Optional: Configure system message and sampling parameters
val conversationConfig = ConversationConfig(
    systemMessage = Message.of("You are a helpful assistant."),
    samplerConfig = SamplerConfig(topK = 10, topP = 0.95, temperature = 0.8),
)

val conversation = engine.createConversation(conversationConfig)
// Or with default config:
// val conversation = engine.createConversation()

// ... Use the conversation ...

// Close the conversation when done
conversation.close()
```

`Conversation` implements `AutoCloseable`, so you can use the `use` block for
automatic resource management for one-shot or short-lived conversation:

```kotlin
engine.createConversation(conversationConfig).use { conversation ->
    // Interact with the conversation
}
```

### 4. Sending Messages

There are three ways to send messages:

-   **`sendMessage(message: Message): Message`**: Synchronous call that blocks
    until the model returns a complete response. This is simpler for basic
    request/response interactions.
-   **`sendMessageAsync(message: Message, callback: MessageCallback)`**:
    Asynchronous call for streaming responses. This is better for long-running
    requests or when you want to display the response as it's being generated.
-   **`sendMessageAsync(message: Message): Flow<Message>`**: Asynchronous call
    that returns a Kotlin Flow for streaming responses. This is the recommended
    approach for Coroutine users.

**Synchronous Example:**

```kotlin
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Message

val userMessage = Message.of("What is the capital of France?")
print(conversation.sendMessage(userMessage))
```

**Asynchronous Example with callback:**

Use `sendMessageAsync` to send a message to the model and receive responses
through callback.

```kotlin
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

val callback = object : MessageCallback {
    override fun onMessage(message: Message) {
        print(message)
    }

    override fun onDone() {
        // Streaming completed
    }

    override fun onError(throwable: Throwable) {
        // Error during streaming
    }
}

val userMessage = Message.of("What is the capital of France?")
conversation.sendMessageAsync(userMessage, callback)
```

**Asynchronous Example with Flow:**

Use `sendMessageAsync` (without the callback arg) to send a message to the model
and receive responses through a Kotlin Flow.

```kotlin
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.launch

// Within a coroutine scope
val userMessage = Message.of("What is the capital of France?")
conversation.sendMessageAsync(userMessage)
    .catch { ... } // error during streaming
    .collect{ print(it.toString()) }
```

### 5. Multi-Modality

Note: this only works with models with multi-modality support. e.g., the
[Gemma3n](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm).

`Message` objects can contain different types of `Content`, including `Text`,
`ImageBytes`, `ImageFile`, and `AudioBytes`, `AudioFile`.

```kotlin
// Initialize the `visionBackend` and/or the `audioBackend`
val engineConfig = EngineConfig(
    modelPath = "/path/to/your/model.litertlm", // Replace with your model path
    backend = Backend.CPU, // Or Backend.GPU
    visionBackend = Backend.GPU,
    audioBackend = Backend.CPU,
)

// See the Content class for other variants.
val multiModalMessage = Message.of(
    Content.ImageFile("/path/to/image"),
    Content.AudioBytes(audioBytes), // ByteArray of the audio
    Content.Text("Describe this image and audio."),
)
```

### 6. Defining and Using Tools

Note: this only works with models with tool support.

You can define custom Kotlin functions as tools that the model can call to
perform actions or fetch information.

#### Defining a ToolSet

Create a class and annotate methods with `@Tool` and parameters with
`@ToolParam`.

```kotlin
import com.google.ai.edge.litertlm.Tool
import com.google.ai.edge.litertlm.ToolParam

class SampleToolSet {
    @Tool(description = "Get the current weather for a city")
    fun getCurrentWeather(
        @ToolParam(description = "The city name, e.g., San Francisco") city: String,
        @ToolParam(description = "Optional country code, e.g., US") country: String? = null,
        @ToolParam(description = "Temperature unit (celsius or fahrenheit). Default: celsius") unit: String = "celsius"
    ): Map<String, Any> {
        // In a real application, you would call a weather API here
        return mapOf("temperature" to 25, "unit" to  unit, "condition" to "Sunny")
    }

    @Tool(description = "Get the sum of a list of numbers.")
    fun sum(
        @ToolParam(description = "The numbers, could be floating point.") numbers: List<Double>,
    ): Double {
        return numbers.sum()
    }
}
```

Behind the scenes, the API inspects these annotations and the function signature
to generate an OpenAPI-style schema. This schema describes the tool's
functionality, parameters (including their types and descriptions from
`@ToolParam`), and return type to the language model.

##### Parameter Types

The types for parameters annotated with `@ToolParam` can be `String`, `Int`,
`Boolean`, `Float`, `Double`, or a `List` of these types (e.g., `List<String>`).
Use nullable types (e.g., `String?`) to indicate nullable parameters. Set a
default value to indicate that the parameter is optional, and mention the
default value in the description in `@ToolParam`.

##### Return Type

The return type of your tool function can be any Kotlin type. The result will be
converted to a JSON element before being sent back to the model.

-   `List` types are converted to JSON array.
-   `Map` types are converted to JSON object.
-   Primitive types (`String`, `Number`, `Boolean`) are converted to the
    corresponding JSON primitive.
-   Other types are converted to string with the `toString()` method.

For structured data, returning `Map` or a data class that will be converted to a
JSON object is recommended.

#### Registering Tools

Include instances of your tool sets in the `ConversationConfig`.

```kotlin
val conversation = engine.createConversation(
    ConversationConfig(
        tools = listOf(SampleToolSet())
        // ... other configs
    )
)

// Send messages that might trigger the tool
val userMessage = Message.of("What's the weather like in London?")
conversation.sendMessageAsync(userMessage, callback)
```

The model will decide when to call the tool based on the conversation. The
results from the tool execution are automatically sent back to the model to
generate the final response.

## Error Handling

API methods can throw `LiteRtLmJniException` for errors from the native layer or
standard Kotlin exceptions like `IllegalStateException` for lifecycle issues.
Always wrap API calls in try-catch blocks. The `onError` callback in
`MessageCallback` will also report errors during asynchronous operations.
