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
include CMakeFiles/wb.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/wb.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/wb.dir/flags.make

CMakeFiles/wb.dir/libwb/wbArg.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbArg.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbArg.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbArg.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbArg.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbArg.cpp

CMakeFiles/wb.dir/libwb/wbArg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbArg.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbArg.cpp > CMakeFiles/wb.dir/libwb/wbArg.cpp.i

CMakeFiles/wb.dir/libwb/wbArg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbArg.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbArg.cpp -o CMakeFiles/wb.dir/libwb/wbArg.cpp.s

CMakeFiles/wb.dir/libwb/wbArg.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbArg.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbArg.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbArg.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbArg.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbArg.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbArg.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbArg.cpp.o

CMakeFiles/wb.dir/libwb/wbMPI.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbMPI.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbMPI.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbMPI.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbMPI.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbMPI.cpp

CMakeFiles/wb.dir/libwb/wbMPI.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbMPI.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbMPI.cpp > CMakeFiles/wb.dir/libwb/wbMPI.cpp.i

CMakeFiles/wb.dir/libwb/wbMPI.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbMPI.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbMPI.cpp -o CMakeFiles/wb.dir/libwb/wbMPI.cpp.s

CMakeFiles/wb.dir/libwb/wbMPI.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbMPI.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbMPI.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbMPI.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbMPI.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbMPI.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbMPI.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbMPI.cpp.o

CMakeFiles/wb.dir/libwb/wbPPM.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbPPM.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbPPM.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbPPM.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbPPM.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbPPM.cpp

CMakeFiles/wb.dir/libwb/wbPPM.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbPPM.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbPPM.cpp > CMakeFiles/wb.dir/libwb/wbPPM.cpp.i

CMakeFiles/wb.dir/libwb/wbPPM.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbPPM.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbPPM.cpp -o CMakeFiles/wb.dir/libwb/wbPPM.cpp.s

CMakeFiles/wb.dir/libwb/wbPPM.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbPPM.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbPPM.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbPPM.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbPPM.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbPPM.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbPPM.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbPPM.cpp.o

CMakeFiles/wb.dir/libwb/wbPath.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbPath.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbPath.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbPath.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbPath.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbPath.cpp

CMakeFiles/wb.dir/libwb/wbPath.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbPath.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbPath.cpp > CMakeFiles/wb.dir/libwb/wbPath.cpp.i

CMakeFiles/wb.dir/libwb/wbPath.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbPath.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbPath.cpp -o CMakeFiles/wb.dir/libwb/wbPath.cpp.s

CMakeFiles/wb.dir/libwb/wbPath.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbPath.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbPath.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbPath.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbPath.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbPath.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbPath.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbPath.cpp.o

CMakeFiles/wb.dir/libwb/wbExport.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbExport.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbExport.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbExport.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbExport.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbExport.cpp

CMakeFiles/wb.dir/libwb/wbExport.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbExport.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbExport.cpp > CMakeFiles/wb.dir/libwb/wbExport.cpp.i

CMakeFiles/wb.dir/libwb/wbExport.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbExport.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbExport.cpp -o CMakeFiles/wb.dir/libwb/wbExport.cpp.s

CMakeFiles/wb.dir/libwb/wbExport.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbExport.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbExport.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbExport.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbExport.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbExport.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbExport.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbExport.cpp.o

CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbDirectory.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbDirectory.cpp

CMakeFiles/wb.dir/libwb/wbDirectory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbDirectory.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbDirectory.cpp > CMakeFiles/wb.dir/libwb/wbDirectory.cpp.i

CMakeFiles/wb.dir/libwb/wbDirectory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbDirectory.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbDirectory.cpp -o CMakeFiles/wb.dir/libwb/wbDirectory.cpp.s

CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o

CMakeFiles/wb.dir/libwb/wbSparse.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbSparse.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbSparse.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbSparse.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbSparse.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbSparse.cpp

CMakeFiles/wb.dir/libwb/wbSparse.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbSparse.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbSparse.cpp > CMakeFiles/wb.dir/libwb/wbSparse.cpp.i

CMakeFiles/wb.dir/libwb/wbSparse.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbSparse.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbSparse.cpp -o CMakeFiles/wb.dir/libwb/wbSparse.cpp.s

CMakeFiles/wb.dir/libwb/wbSparse.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbSparse.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbSparse.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbSparse.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbSparse.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbSparse.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbSparse.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbSparse.cpp.o

CMakeFiles/wb.dir/libwb/wbTimer.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbTimer.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbTimer.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_8)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbTimer.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbTimer.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbTimer.cpp

CMakeFiles/wb.dir/libwb/wbTimer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbTimer.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbTimer.cpp > CMakeFiles/wb.dir/libwb/wbTimer.cpp.i

CMakeFiles/wb.dir/libwb/wbTimer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbTimer.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbTimer.cpp -o CMakeFiles/wb.dir/libwb/wbTimer.cpp.s

CMakeFiles/wb.dir/libwb/wbTimer.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbTimer.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbTimer.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbTimer.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbTimer.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbTimer.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbTimer.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbTimer.cpp.o

CMakeFiles/wb.dir/libwb/wbFile.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbFile.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbFile.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_9)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbFile.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbFile.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbFile.cpp

CMakeFiles/wb.dir/libwb/wbFile.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbFile.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbFile.cpp > CMakeFiles/wb.dir/libwb/wbFile.cpp.i

CMakeFiles/wb.dir/libwb/wbFile.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbFile.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbFile.cpp -o CMakeFiles/wb.dir/libwb/wbFile.cpp.s

CMakeFiles/wb.dir/libwb/wbFile.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbFile.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbFile.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbFile.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbFile.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbFile.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbFile.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbFile.cpp.o

CMakeFiles/wb.dir/libwb/wbInit.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbInit.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbInit.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_10)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbInit.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbInit.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbInit.cpp

CMakeFiles/wb.dir/libwb/wbInit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbInit.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbInit.cpp > CMakeFiles/wb.dir/libwb/wbInit.cpp.i

CMakeFiles/wb.dir/libwb/wbInit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbInit.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbInit.cpp -o CMakeFiles/wb.dir/libwb/wbInit.cpp.s

CMakeFiles/wb.dir/libwb/wbInit.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbInit.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbInit.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbInit.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbInit.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbInit.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbInit.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbInit.cpp.o

CMakeFiles/wb.dir/libwb/wbDataset.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbDataset.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbDataset.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_11)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbDataset.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbDataset.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbDataset.cpp

CMakeFiles/wb.dir/libwb/wbDataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbDataset.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbDataset.cpp > CMakeFiles/wb.dir/libwb/wbDataset.cpp.i

CMakeFiles/wb.dir/libwb/wbDataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbDataset.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbDataset.cpp -o CMakeFiles/wb.dir/libwb/wbDataset.cpp.s

CMakeFiles/wb.dir/libwb/wbDataset.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbDataset.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbDataset.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbDataset.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbDataset.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbDataset.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbDataset.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbDataset.cpp.o

CMakeFiles/wb.dir/libwb/wbExit.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbExit.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbExit.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_12)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbExit.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbExit.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbExit.cpp

CMakeFiles/wb.dir/libwb/wbExit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbExit.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbExit.cpp > CMakeFiles/wb.dir/libwb/wbExit.cpp.i

CMakeFiles/wb.dir/libwb/wbExit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbExit.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbExit.cpp -o CMakeFiles/wb.dir/libwb/wbExit.cpp.s

CMakeFiles/wb.dir/libwb/wbExit.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbExit.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbExit.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbExit.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbExit.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbExit.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbExit.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbExit.cpp.o

CMakeFiles/wb.dir/libwb/wbSolution.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbSolution.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbSolution.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_13)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbSolution.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbSolution.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbSolution.cpp

CMakeFiles/wb.dir/libwb/wbSolution.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbSolution.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbSolution.cpp > CMakeFiles/wb.dir/libwb/wbSolution.cpp.i

CMakeFiles/wb.dir/libwb/wbSolution.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbSolution.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbSolution.cpp -o CMakeFiles/wb.dir/libwb/wbSolution.cpp.s

CMakeFiles/wb.dir/libwb/wbSolution.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbSolution.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbSolution.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbSolution.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbSolution.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbSolution.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbSolution.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbSolution.cpp.o

CMakeFiles/wb.dir/libwb/wbLogger.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbLogger.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbLogger.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_14)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbLogger.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbLogger.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbLogger.cpp

CMakeFiles/wb.dir/libwb/wbLogger.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbLogger.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbLogger.cpp > CMakeFiles/wb.dir/libwb/wbLogger.cpp.i

CMakeFiles/wb.dir/libwb/wbLogger.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbLogger.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbLogger.cpp -o CMakeFiles/wb.dir/libwb/wbLogger.cpp.s

CMakeFiles/wb.dir/libwb/wbLogger.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbLogger.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbLogger.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbLogger.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbLogger.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbLogger.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbLogger.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbLogger.cpp.o

CMakeFiles/wb.dir/libwb/wbImage.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbImage.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbImage.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_15)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbImage.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbImage.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbImage.cpp

CMakeFiles/wb.dir/libwb/wbImage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbImage.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbImage.cpp > CMakeFiles/wb.dir/libwb/wbImage.cpp.i

CMakeFiles/wb.dir/libwb/wbImage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbImage.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbImage.cpp -o CMakeFiles/wb.dir/libwb/wbImage.cpp.s

CMakeFiles/wb.dir/libwb/wbImage.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbImage.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbImage.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbImage.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbImage.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbImage.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbImage.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbImage.cpp.o

CMakeFiles/wb.dir/libwb/wbImport.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbImport.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbImport.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_16)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbImport.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbImport.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbImport.cpp

CMakeFiles/wb.dir/libwb/wbImport.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbImport.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbImport.cpp > CMakeFiles/wb.dir/libwb/wbImport.cpp.i

CMakeFiles/wb.dir/libwb/wbImport.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbImport.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbImport.cpp -o CMakeFiles/wb.dir/libwb/wbImport.cpp.s

CMakeFiles/wb.dir/libwb/wbImport.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbImport.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbImport.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbImport.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbImport.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbImport.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbImport.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbImport.cpp.o

CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbCUDA.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_17)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbCUDA.cpp

CMakeFiles/wb.dir/libwb/wbCUDA.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbCUDA.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbCUDA.cpp > CMakeFiles/wb.dir/libwb/wbCUDA.cpp.i

CMakeFiles/wb.dir/libwb/wbCUDA.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbCUDA.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbCUDA.cpp -o CMakeFiles/wb.dir/libwb/wbCUDA.cpp.s

CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o

CMakeFiles/wb.dir/libwb/wbUtils.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/wbUtils.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/wbUtils.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_18)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/wbUtils.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/wbUtils.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/wbUtils.cpp

CMakeFiles/wb.dir/libwb/wbUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/wbUtils.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/wbUtils.cpp > CMakeFiles/wb.dir/libwb/wbUtils.cpp.i

CMakeFiles/wb.dir/libwb/wbUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/wbUtils.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/wbUtils.cpp -o CMakeFiles/wb.dir/libwb/wbUtils.cpp.s

CMakeFiles/wb.dir/libwb/wbUtils.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/wbUtils.cpp.o.requires

CMakeFiles/wb.dir/libwb/wbUtils.cpp.o.provides: CMakeFiles/wb.dir/libwb/wbUtils.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/wbUtils.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/wbUtils.cpp.o.provides

CMakeFiles/wb.dir/libwb/wbUtils.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/wbUtils.cpp.o

CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o: CMakeFiles/wb.dir/flags.make
CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o: /home/u32/gmrosier/ece569/labs/libwb/vendor/json11.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/u32/gmrosier/ece569/build_dir/CMakeFiles $(CMAKE_PROGRESS_19)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o -c /home/u32/gmrosier/ece569/labs/libwb/vendor/json11.cpp

CMakeFiles/wb.dir/libwb/vendor/json11.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wb.dir/libwb/vendor/json11.cpp.i"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/u32/gmrosier/ece569/labs/libwb/vendor/json11.cpp > CMakeFiles/wb.dir/libwb/vendor/json11.cpp.i

CMakeFiles/wb.dir/libwb/vendor/json11.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wb.dir/libwb/vendor/json11.cpp.s"
	/opt/rh/devtoolset-1.1/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/u32/gmrosier/ece569/labs/libwb/vendor/json11.cpp -o CMakeFiles/wb.dir/libwb/vendor/json11.cpp.s

CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o.requires:
.PHONY : CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o.requires

CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o.provides: CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o.requires
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o.provides.build
.PHONY : CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o.provides

CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o.provides.build: CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o

# Object files for target wb
wb_OBJECTS = \
"CMakeFiles/wb.dir/libwb/wbArg.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbMPI.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbPPM.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbPath.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbExport.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbSparse.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbTimer.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbFile.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbInit.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbDataset.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbExit.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbSolution.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbLogger.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbImage.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbImport.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o" \
"CMakeFiles/wb.dir/libwb/wbUtils.cpp.o" \
"CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o"

# External object files for target wb
wb_EXTERNAL_OBJECTS =

libwb.a: CMakeFiles/wb.dir/libwb/wbArg.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbMPI.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbPPM.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbPath.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbExport.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbSparse.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbTimer.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbFile.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbInit.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbDataset.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbExit.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbSolution.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbLogger.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbImage.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbImport.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/wbUtils.cpp.o
libwb.a: CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o
libwb.a: CMakeFiles/wb.dir/build.make
libwb.a: CMakeFiles/wb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libwb.a"
	$(CMAKE_COMMAND) -P CMakeFiles/wb.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/wb.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/wb.dir/build: libwb.a
.PHONY : CMakeFiles/wb.dir/build

CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbArg.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbMPI.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbPPM.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbPath.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbExport.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbDirectory.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbSparse.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbTimer.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbFile.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbInit.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbDataset.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbExit.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbSolution.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbLogger.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbImage.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbImport.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbCUDA.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/wbUtils.cpp.o.requires
CMakeFiles/wb.dir/requires: CMakeFiles/wb.dir/libwb/vendor/json11.cpp.o.requires
.PHONY : CMakeFiles/wb.dir/requires

CMakeFiles/wb.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/wb.dir/cmake_clean.cmake
.PHONY : CMakeFiles/wb.dir/clean

CMakeFiles/wb.dir/depend:
	cd /home/u32/gmrosier/ece569/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/u32/gmrosier/ece569/labs /home/u32/gmrosier/ece569/labs /home/u32/gmrosier/ece569/build_dir /home/u32/gmrosier/ece569/build_dir /home/u32/gmrosier/ece569/build_dir/CMakeFiles/wb.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/wb.dir/depend

