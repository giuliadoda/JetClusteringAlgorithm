# USAGE :))
# make openmp          			# compile ALL openmp versions 
# make bin/openmp_version1   	# compile only specific openmp version
# make all             			# serial + all openmp

# =========================
# Compiler
# =========================

CC = gcc


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
HDF5_LIBS   = $(shell pkg-config --libs hdf5)


# =========================
# Include paths
# =========================

INCLUDES = -I$(INC_DIR)

SERIAL_INCLUDES = -I$(INC_DIR)/serial
OMP_INCLUDES    = -I$(INC_DIR)/openmp


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

rebuild: clean serial openmp

all: serial openmp


# automatic header dependencies
-include $(SERIAL_OBJ:.o=.d)
-include $(OMP_ALL_OBJ:.o=.d)
-include $(OMP_COMMON_OBJ:.o=.d)


.PHONY: serial openmp all clean rebuild