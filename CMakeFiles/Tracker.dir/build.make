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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ashok/fuerte_workspace/ecp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ashok/fuerte_workspace/ecp

# Include any dependencies generated for this target.
include CMakeFiles/Tracker.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Tracker.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Tracker.dir/flags.make

CMakeFiles/Tracker.dir/src/surfDestTracker.o: CMakeFiles/Tracker.dir/flags.make
CMakeFiles/Tracker.dir/src/surfDestTracker.o: src/surfDestTracker.cpp
CMakeFiles/Tracker.dir/src/surfDestTracker.o: manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/share/geometry_msgs/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/share/sensor_msgs/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/share/ros/core/rosbuild/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/share/roslib/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/share/rosconsole/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/pluginlib/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/share/message_filters/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/share/roslang/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/share/roscpp/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/image_common/image_transport/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/share/std_msgs/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/vision_opencv/opencv2/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/vision_opencv/cv_bridge/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/common_rosdeps/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/image_common/camera_calibration_parsers/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/share/rostest/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/image_common/camera_info_manager/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/share/rospy/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/bond_core/bond/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/bond_core/smclib/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/bond_core/bondcpp/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/nodelet_core/nodelet/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/camera_umd/uvc_camera/manifest.xml
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/bond_core/bond/msg_gen/generated
CMakeFiles/Tracker.dir/src/surfDestTracker.o: /opt/ros/fuerte/stacks/nodelet_core/nodelet/srv_gen/generated
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ashok/fuerte_workspace/ecp/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Tracker.dir/src/surfDestTracker.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -W -Wall -Wno-unused-parameter -fno-strict-aliasing -pthread -o CMakeFiles/Tracker.dir/src/surfDestTracker.o -c /home/ashok/fuerte_workspace/ecp/src/surfDestTracker.cpp

CMakeFiles/Tracker.dir/src/surfDestTracker.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Tracker.dir/src/surfDestTracker.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -W -Wall -Wno-unused-parameter -fno-strict-aliasing -pthread -E /home/ashok/fuerte_workspace/ecp/src/surfDestTracker.cpp > CMakeFiles/Tracker.dir/src/surfDestTracker.i

CMakeFiles/Tracker.dir/src/surfDestTracker.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Tracker.dir/src/surfDestTracker.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -W -Wall -Wno-unused-parameter -fno-strict-aliasing -pthread -S /home/ashok/fuerte_workspace/ecp/src/surfDestTracker.cpp -o CMakeFiles/Tracker.dir/src/surfDestTracker.s

CMakeFiles/Tracker.dir/src/surfDestTracker.o.requires:
.PHONY : CMakeFiles/Tracker.dir/src/surfDestTracker.o.requires

CMakeFiles/Tracker.dir/src/surfDestTracker.o.provides: CMakeFiles/Tracker.dir/src/surfDestTracker.o.requires
	$(MAKE) -f CMakeFiles/Tracker.dir/build.make CMakeFiles/Tracker.dir/src/surfDestTracker.o.provides.build
.PHONY : CMakeFiles/Tracker.dir/src/surfDestTracker.o.provides

CMakeFiles/Tracker.dir/src/surfDestTracker.o.provides.build: CMakeFiles/Tracker.dir/src/surfDestTracker.o

# Object files for target Tracker
Tracker_OBJECTS = \
"CMakeFiles/Tracker.dir/src/surfDestTracker.o"

# External object files for target Tracker
Tracker_EXTERNAL_OBJECTS =

bin/Tracker: CMakeFiles/Tracker.dir/src/surfDestTracker.o
bin/Tracker: CMakeFiles/Tracker.dir/build.make
bin/Tracker: CMakeFiles/Tracker.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bin/Tracker"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Tracker.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Tracker.dir/build: bin/Tracker
.PHONY : CMakeFiles/Tracker.dir/build

CMakeFiles/Tracker.dir/requires: CMakeFiles/Tracker.dir/src/surfDestTracker.o.requires
.PHONY : CMakeFiles/Tracker.dir/requires

CMakeFiles/Tracker.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Tracker.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Tracker.dir/clean

CMakeFiles/Tracker.dir/depend:
	cd /home/ashok/fuerte_workspace/ecp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ashok/fuerte_workspace/ecp /home/ashok/fuerte_workspace/ecp /home/ashok/fuerte_workspace/ecp /home/ashok/fuerte_workspace/ecp /home/ashok/fuerte_workspace/ecp/CMakeFiles/Tracker.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Tracker.dir/depend
