
# This makefile requires the following variables:
# CXX
# CXXFLAGS
# CXXFLAGS_DEBUG
# NVCC
# CUDAFLAGS
# INCLUDES
# LDFLAGS
# LDLIBS

ifndef SYSTEM
$(error SYSTEM variable is not set)
endif
ifeq ($(wildcard sys/$(SYSTEM)/.),)
$(error $(SYSTEM) is not an available system. Should be one of: $(shell ls sys))
endif

CXXFLAGS+= -Xcompiler -Wall -std=c++11 -g -Xcompiler -fopenmp -O3
CXXFLAGS_DEBUG+= -Xcompiler -Wall -std=c++11 -g -Xcompiler -fopenmp -O0 -DDEBUG

CUDAFLAGS+=-m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60

INCLUDES+=-I./include -I./ -I./sys/$(SYSTEM)/

LDLIBS+=-lboost_unit_test_framework -lboost_log -lcublas -lcusolver -lcusparse -lboost_program_options -lconfig++

TESTS:=$(wildcard tests/*.cpp)
TESTOBJS:=$(addprefix objects/$(SYSTEM)/,$(TESTS:.cpp=.o))

SYS:=$(wildcard sys/$(SYSTEM)/*.cpp)
SYSNOTDIR:=$(notdir $(SYS))
SYSOBJS:=$(addprefix objects/$(SYSTEM)/sys/,$(SYSNOTDIR:.cpp=.o))
SYSOBJS_DEBUG:=$(addprefix objects/$(SYSTEM)/sys/,$(SYSNOTDIR:.cpp=_debug.o))

SRCS:=$(wildcard src/*.cpp)
OBJS:=$(addprefix objects/$(SYSTEM)/,$(SRCS:.cpp=.o))
OBJS_DEBUG:=$(addprefix objects/$(SYSTEM)/,$(SRCS:.cpp=_debug.o))

DEP:=$(subst .o,.d,$(OBJS)) $(subst .o,.d,$(OBJS_DEBUG)) $(subst .o,.d,$(TESTOBJS)) $(subst .o,.d,$(SYSOBJS)) $(subst .o,.d,$(SYSOBJS_DEBUG))

.PHONY: default
default: $(SYSTEM)

# Compile src files
objects/$(SYSTEM)/src/%.o: src/%.cpp | object-dirs
	@echo -n "Compiling $<..."	
	@$(NVCC) $(INCLUDES) $(CXXFLAGS) -DBOOST_LOG_DYN_LINK -DBOOST_TEST_DYN_LINK $(CUDAFLAGS) -c $< -o $@
	@echo " done"

# Compile debug system src files
objects/$(SYSTEM)/sys/%_debug.o: sys/$(SYSTEM)/%.cpp | object-dirs
	@echo -n "Compiling $< (debug)..."	
	@$(NVCC) $(INCLUDES) $(CXXFLAGS_DEBUG) -DBOOST_LOG_DYN_LINK -DBOOST_TEST_DYN_LINK $(CUDAFLAGS) -c $< -o $@
	@echo " done"

# Compile src system files
objects/$(SYSTEM)/sys/%.o: sys/$(SYSTEM)/%.cpp | object-dirs
	@echo -n "Compiling $<..."	
	@$(NVCC) $(INCLUDES) $(CXXFLAGS) -DBOOST_LOG_DYN_LINK -DBOOST_TEST_DYN_LINK $(CUDAFLAGS) -c $< -o $@
	@echo " done"

# Compile debug src files
objects/$(SYSTEM)/src/%_debug.o: src/%.cpp | object-dirs
	@echo -n "Compiling $< (debug)..."	
	@$(NVCC) $(INCLUDES) $(CXXFLAGS_DEBUG) -DBOOST_LOG_DYN_LINK -DBOOST_TEST_DYN_LINK $(CUDAFLAGS) -c $< -o $@
	@echo " done"

# Compile unit tests
objects/$(SYSTEM)/tests/%.o: tests/%.cpp | object-dirs
	@echo -n "Compiling $<..."	
	@$(NVCC) $(INCLUDES) $(CXXFLAGS) -DBOOST_LOG_DYN_LINK -DBOOST_TEST_DYN_LINK $(CUDAFLAGS) -c $< -o $@
	@echo " done"

# Link unit tests and run
.PHONY: unit-tests
unit-tests: $(SYSTEM)-unit-tests-run

# Run the unit tests
.PHONY: $(SYSTEM)-unit-tests-run
$(SYSTEM)-unit-tests-run : $(SYSTEM)-unit-tests
	@./$(SYSTEM)-unit-tests

# Compile the unit tests
$(SYSTEM)-unit-tests: $(TESTOBJS) $(OBJS) $(SYSOBJS) | object-dirs
	@echo  -n "Linking $@..."
	@$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) $(TESTOBJS) $(SYSOBJS) $(filter-out objects/$(SYSTEM)/src/main.o, $(OBJS)) $(LDFLAGS) $(LDLIBS) -o $@
	@echo " done"

# Link program
$(SYSTEM): $(OBJS) $(SYSOBJS) | object-dirs
	@echo  -n "Linking $@..."
	@$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) $(OBJS) $(SYSOBJS) $(LDFLAGS) $(LDLIBS) -o $@
	@echo " done"

.PHONY: debug
debug: $(SYSTEM)-debug

# Link program debug
$(SYSTEM)-debug: $(OBJS_DEBUG) $(SYSOBJS_DEBUG) | object-dirs
	@echo  -n "Linking $@..."
	@$(NVCC) $(CXXFLAGS_DEBUG) $(CUDAFLAGS) $(OBJS_DEBUG) $(SYSOBJS_DEBUG) $(LDFLAGS) $(LDLIBS) -o $@
	@echo " done"

# Generate dependencies for tests
objects/$(SYSTEM)/tests/%.d: tests/%.cpp | object-dirs
	@echo -n 'Generating dependencies for $<...'
	@echo -n $(dir $@) > $@
	@$(CXX) -MM -MP $(INCLUDES) $< $(ALLINC) >> $@
	@echo " done"

# Generate dependencies for src files
objects/$(SYSTEM)/src/%.d: src/%.cpp | object-dirs
	@echo -n 'Generating dependencies for $<...'
	@echo -n $(dir $@) > $@
	@$(CXX) -MM -MP $(INCLUDES) $< $(ALLINC) >> $@
	@cp $@ $(subst .d,_debug.d,$@)
	@sed -i 's|$(subst .d,.o,$@):\(.*\)|$(subst .d,_debug.o,$@):\1|g' $(subst .d,_debug.d,$@) 
	@echo " done"

# Generate dependencies for sys src files
objects/$(SYSTEM)/sys/%.d: sys/$(SYSTEM)/%.cpp | object-dirs
	@echo -n 'Generating dependencies for $<...'
	@echo -n $(dir $@) > $@
	@$(CXX) -MM -MP $(INCLUDES) $< $(ALLINC) >> $@
	@echo " done"

# Capture current git commit. 
# Phony so it's updated every time
.PHONY: githash.h
githash.h:
	@echo "Generating GIT hash file..."
	@printf '#ifndef GIT_HASH\n#define GIT_HASH "' > $@ && \
	git rev-parse HEAD | tr -d "\n" >> $@ && \
	printf '"\n#endif' >> $@

# Compile everything
.PHONY: all
all: unit-tests debug default

# Run all tests
.PHONY: test
test: unit-tests regression-tests

# Make and run regression tests
.PHONY: regression-tests
regression-tests: $(SYSTEM)
	@echo "Running regression tests..."
	@./regression-tests

# Create directories
.PHONY: object-dirs
object-dirs:
	@mkdir -p objects/$(SYSTEM)/tests
	@mkdir -p objects/$(SYSTEM)/src
	@mkdir -p objects/$(SYSTEM)/sys

# Include generated dependencies
-include $(DEP)

# Remove object files
.PHONY: clean
clean:
	rm -f objects/$(SYSTEM)/tests/*.o
	rm -f objects/$(SYSTEM)/src/*.o
	rm -f objects/$(SYSTEM)/sys/*.o

# Remove everything that can be regenerated
.PHONY: distclean
distclean: clean
	rm -rf objects
	rm -f regressionTest.lock
	rm -f $(SYSTEM)
	rm -f $(SYSTEM)-debug
	rm -f $(SYSTEM)-unit-tests

.PHONY: list
list:
	@echo "tests: $(TESTS)"
	@echo "test objects: $(TESTOBJS)"
	@echo "sources: $(SRCS)"
	@echo "system: $(SYS)"
	@echo "objects: $(OBJS) $(SYSOBJS)"
	@echo "dependencies: $(DEP)"

.PHONY: help
help:
	@echo "all              - Compile everything"
	@echo "unit-tests       - Compile and run unit tests"
	@echo "regression-tests - Compile and run regression tests"
	@echo "test             - Compile and run all tests"
	@echo "default          - Compile program"
	@echo "debug            - Compile program (with O0 and DEBUG defined)"
	@echo "clean            - Remove object files"
	@echo "distclean        - Remove all objects and dependencies"

