# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/u32/gmrosier/ece569/labs

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/u32/gmrosier/ece569/build_dir

# Include any dependencies generated for this target.
include CMakeFiles/DeviceQuery_Solution.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/DeviceQuery_Solution.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DeviceQuery_Solution.dir/flags.make

CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/./DeviceQuery_Solution_generated_template.cu.o: CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/DeviceQuery_Solution_generated_template.cu.o.depend
CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/./DeviceQuery_Solution_generated_template.cu.o: CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/DeviceQuery_Solution_generated_template.cu.o.cmake
CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/./DeviceQuery_Solution_generated_template.cu.o: /home/u32/gmrosier/ece569/labs/Module2/DeviceQuery/template.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/./DeviceQuery_Solution_generated_template.cu.o"
	cd /home/u32/gmrosier/ece569/build_dir/CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery && /usr/bin/cmake -E make_directory /home/u32/gmrosier/ece569/build_dir/CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/.
	cd /home/u32/gmrosier/ece569/build_dir/CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/home/u32/gmrosier/ece569/build_dir/CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/./DeviceQuery_Solution_generated_template.cu.o -D generated_cubin_file:STRING=/home/u32/gmrosier/ece569/build_dir/CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/./DeviceQuery_Solution_generated_template.cu.o.cubin.txt -P /home/u32/gmrosier/ece569/build_dir/CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/DeviceQuery_Solution_generated_template.cu.o.cmake

# Object files for target DeviceQuery_Solution
DeviceQuery_Solution_OBJECTS =

# External object files for target DeviceQuery_Solution
DeviceQuery_Solution_EXTERNAL_OBJECTS = \
"/home/u32/gmrosier/ece569/build_dir/CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/./DeviceQuery_Solution_generated_template.cu.o"

DeviceQuery_Solution: CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/./DeviceQuery_Solution_generated_template.cu.o
DeviceQuery_Solution: CMakeFiles/DeviceQuery_Solution.dir/build.make
DeviceQuery_Solution: /uaopt/cuda/7.0.28/lib64/libcudart.so
DeviceQuery_Solution: libwb.a
DeviceQuery_Solution: /uaopt/cuda/7.0.28/lib64/libcudart.so
DeviceQuery_Solution: CMakeFiles/DeviceQuery_Solution.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable DeviceQuery_Solution"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DeviceQuery_Solution.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DeviceQuery_Solution.dir/build: DeviceQuery_Solution
.PHONY : CMakeFiles/DeviceQuery_Solution.dir/build

CMakeFiles/DeviceQuery_Solution.dir/requires:
.PHONY : CMakeFiles/DeviceQuery_Solution.dir/requires

CMakeFiles/DeviceQuery_Solution.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DeviceQuery_Solution.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DeviceQuery_Solution.dir/clean

CMakeFiles/DeviceQuery_Solution.dir/depend: CMakeFiles/DeviceQuery_Solution.dir/Module2/DeviceQuery/./DeviceQuery_Solution_generated_template.cu.o
	cd /home/u32/gmrosier/ece569/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/u32/gmrosier/ece569/labs /home/u32/gmrosier/ece569/labs /home/u32/gmrosier/ece569/build_dir /home/u32/gmrosier/ece569/build_dir /home/u32/gmrosier/ece569/build_dir/CMakeFiles/DeviceQuery_Solution.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DeviceQuery_Solution.dir/depend

