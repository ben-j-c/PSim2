# Purpose
Playing around with CUDA and making some interesting simulations.

# Demo
Here I make a galaxy from a box of 10,000 particles.
![Demo](https://github.com/ben-j-c/PSim2/blob/master/PSIM_demo.gif?raw=true "Just a demo.")

# Compilation Dependencies
* [GLFW 3.3.2](https://www.glfw.org/download.html)
  * Environment variable: `GLFW_x64` is the root directory of includes and libraries. i.e., `$(GLFW_x64)/lib` should resolve to the library and `$(GLFW_x64)/include` for the headers.
* [GLEW](http://glew.sourceforge.net/)
  * Environment variable: `GLEW_PATH` is the root directory of includes and libraries. i.e., `$(GLEW_PATH)/lib/Release/x64` and `$(GLEW_PATH)/include` blah blah.
* Visual Studio 2015/17/18
* CUDA 10.2

# Compilation
1. Install Visual Studio and CUDA.
2. Download GLFW and GLEW
3. Clone.
4. Open PSim.sln and build either Debug or Release

# GUI
The GUI was developed using the OpenGL3 + GLFW implementation for [ImGui](https://github.com/ocornut/imgui). 

# Future Feature Goals
* Fully parameterizing initial conditions for gravity simulation.
* Fast multipole method implementation for gravity.
* Possibly interacting particles (merging, annihilation, collision, etc.).
* Simulations other than just gravity.

# Future Development Goals
* Rewriting kernels to use C++.
* Fixing memory issues.
* Refactoring functions out of kernel.cu and moving them to Model.cpp.
* Abstract Model to use inheritance to allow multiple different simulations.
* Properly fix memory errors.
