#include "imgui_app.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include <GLFW/glfw3.h>

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <filesystem>

namespace chat {

struct ImGuiApp::Impl {
  GLFWwindow* window = nullptr;
  std::atomic<bool> exit_requested{false};
  int width = 1280;
  int height = 720;
};

static void GlfwErrorCallback(int error, const char* description) {
  std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

ImGuiApp::ImGuiApp(const ImGuiAppConfig& config) : impl_(std::make_unique<Impl>()) {
  impl_->width = config.width;
  impl_->height = config.height;

  // Initialize GLFW
  glfwSetErrorCallback(GlfwErrorCallback);
  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  // GL 3.2 + GLSL 150 (macOS requirement)
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  // Create window
  impl_->window = glfwCreateWindow(config.width, config.height, config.title.c_str(), nullptr, nullptr);
  if (!impl_->window) {
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window");
  }

  glfwMakeContextCurrent(impl_->window);
  if (config.vsync) {
    glfwSwapInterval(1);
  } else {
    glfwSwapInterval(0);
  }

  // Initialize ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  // Setup style
  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 4.0f;
  style.FrameRounding = 2.0f;
  style.ScrollbarRounding = 2.0f;

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(impl_->window, true);
  ImGui_ImplOpenGL3_Init("#version 150");

  // Load a readable system font (Menlo) on macOS
  // Fallback to default if not found
  const char* font_path = "/System/Library/Fonts/Menlo.ttc";
  const float font_size = 16.0f;
  
  // Use std::filesystem to check existence (C++17)
  bool font_loaded = false;
  if (std::filesystem::exists(font_path)) {
    ImFont* font = io.Fonts->AddFontFromFileTTF(font_path, font_size);
    if (font) {
      font_loaded = true;
      std::cout << "Loaded font: " << font_path << " (" << font_size << "px)" << std::endl;
    }
  }

  if (!font_loaded) {
    std::cout << "Using default font (could not load " << font_path << ")" << std::endl;
    // Load default font but scale it if possible? Default is bitmap, scaling looks bad.
    // Just add it as is.
    io.Fonts->AddFontDefault();
  }

  std::cout << "ImGui application initialized: " << config.width << "x" << config.height << std::endl;
}

ImGuiApp::~ImGuiApp() {
  if (impl_->window) {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(impl_->window);
    glfwTerminate();
  }
}

void ImGuiApp::Run(std::function<void()> render_fn) {
  while (!glfwWindowShouldClose(impl_->window) && !impl_->exit_requested.load()) {
    glfwPollEvents();

    // Handle window resize
    int width, height;
    glfwGetFramebufferSize(impl_->window, &width, &height);
    impl_->width = width;
    impl_->height = height;

    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Create a fullscreen dockspace
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                    ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                                    ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpaceWindow", nullptr, window_flags);
    ImGui::PopStyleVar(3);

    // Call the user's render function
    if (render_fn) {
      render_fn();
    }

    ImGui::End();

    // Render
    ImGui::Render();
    glViewport(0, 0, width, height);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(impl_->window);
  }
}

void ImGuiApp::RequestExit() {
  impl_->exit_requested.store(true);
}

bool ImGuiApp::ShouldClose() const {
  return glfwWindowShouldClose(impl_->window) || impl_->exit_requested.load();
}

int ImGuiApp::GetWidth() const {
  return impl_->width;
}

int ImGuiApp::GetHeight() const {
  return impl_->height;
}

}  // namespace chat
