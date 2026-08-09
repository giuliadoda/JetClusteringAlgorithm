# USAGE :))
# make openmp          			# compile ALL openmp versions 
# make bin/openmp_version1   	# compile only specific openmp version
# make all             			# serial + all openmp 
# make cuda EXTRA_DEFS="-DTHR_BLOCK=128 -DN_EVENTS=5000"	# CUDA, make clean before each CUDA compilation

# =========================
# Compiler
# =========================

CC = gcc
NVCC = nvcc

# =========================
# Flags
# =========================

CFLAGS = -Wall -Wextra -O0 -MMD -MP
OMP_FLAGS = -fopenmp

# =========================
# Directories
# =========================

SRC_DIR = source
INC_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# =========================
# HDF5
# =========================

HDF5_CFLAGS = $(shell pkg-config --cflags hdf5)
HDF5_LIBS = $(shell pkg-config --libs hdf5)

# =========================
# Include paths
# =========================

INCLUDES = -I$(INC_DIR)

SERIAL_INCLUDES = -I$(INC_DIR)/serial
OMP_INCLUDES = -I$(INC_DIR)/openmp
CU_INCLUDES = -I$(INC_DIR)/cuda

# =========================
# SERIAL VERSION
# =========================

SERIAL_SRC = \
	$(SRC_DIR)/serial/serial_version.c \
	$(SRC_DIR)/serial/functions.c

SERIAL_OBJ = $(SERIAL_SRC:%.c=$(BUILD_DIR)/%.o)

$(SERIAL_OBJ): INCLUDES += $(SERIAL_INCLUDES)

serial: $(BIN_DIR)/serial_version

$(BIN_DIR)/serial_version: $(SERIAL_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CC) $(SERIAL_OBJ) -o $@ $(HDF5_LIBS)


# =========================
# OPENMP VERSIONS
# =========================

# list every openmp "main" version here (without .c)
OMP_VERSIONS = openmp_version_base openmp_version_base_schedule

# functions.c shared by all openmp versions
OMP_COMMON_SRC = $(SRC_DIR)/openmp/functions.c
OMP_COMMON_OBJ = $(OMP_COMMON_SRC:%.c=$(BUILD_DIR)/%.o)

$(OMP_COMMON_OBJ): INCLUDES += $(OMP_INCLUDES)
$(OMP_COMMON_OBJ): CFLAGS   += $(OMP_FLAGS)

# template: for each version name $(1), generate object + binary rules
define OMP_TEMPLATE

$(BUILD_DIR)/$(SRC_DIR)/openmp/$(1).o: INCLUDES += $(OMP_INCLUDES)
$(BUILD_DIR)/$(SRC_DIR)/openmp/$(1).o: CFLAGS   += $(OMP_FLAGS)

$(BIN_DIR)/$(1): $(BUILD_DIR)/$(SRC_DIR)/openmp/$(1).o $(OMP_COMMON_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OMP_FLAGS) $$^ -o $$@ $(HDF5_LIBS)

openmp: $(BIN_DIR)/$(1)

OMP_ALL_OBJ += $(BUILD_DIR)/$(SRC_DIR)/openmp/$(1).o

endef

$(foreach v,$(OMP_VERSIONS),$(eval $(call OMP_TEMPLATE,$(v))))


## =========================
# CUDA VERSION
# =========================

CUDA_MAIN = cuda_version

CUDA_SRC = \
	$(SRC_DIR)/cuda/cuda_version.cu \
	$(SRC_DIR)/cuda/functions.cu

CUDA_OBJ = $(CUDA_SRC:%.cu=$(BUILD_DIR)/%.o)

EXTRA_DEFS ?=
NVFLAGS = -O0 -MMD -MP $(EXTRA_DEFS)

$(CUDA_OBJ): INCLUDES += $(CU_INCLUDES)

cuda: $(BIN_DIR)/$(CUDA_MAIN)

$(BIN_DIR)/$(CUDA_MAIN): $(CUDA_OBJ)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CUDA_OBJ) -o $@ $(HDF5_LIBS)

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) $(HDF5_CFLAGS) $(INCLUDES) -c $< -o $@

# =========================
# Compile rule
# =========================

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(HDF5_CFLAGS) $(INCLUDES) -c $< -o $@


# =========================
# Utilities
# =========================

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

all: serial openmp cuda

rebuild: clean all

# automatic header dependencies
-include $(SERIAL_OBJ:.o=.d)
-include $(OMP_ALL_OBJ:.o=.d)
-include $(OMP_COMMON_OBJ:.o=.d)
-include $(CUDA_OBJ:.o=.d)


.PHONY: serial openmp cuda clean rebuild all