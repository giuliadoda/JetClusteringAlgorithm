# =========================
# Compiler
# =========================

CC = gcc


# =========================
# Flags
# =========================

# CFLAGS = -Wall -Wextra -O2 -MMD -MP
CFLAGS = -Wall -Wextra -O0 -MMD -MP


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

INCLUDES = \
	-I$(INC_DIR) \
	-I$(INC_DIR)/serial


# =========================
# SERIAL VERSION
# =========================

SERIAL_SRC = \
	$(SRC_DIR)/serial/serial_version.c \
	$(SRC_DIR)/serial/functions.c


SERIAL_OBJ = $(SERIAL_SRC:%.c=$(BUILD_DIR)/%.o)


serial: $(BIN_DIR)/serial_version


$(BIN_DIR)/serial_version: $(SERIAL_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CC) $(SERIAL_OBJ) -o $@ $(HDF5_LIBS)


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


rebuild: clean serial


# automatic header dependencies
-include $(SERIAL_OBJ:.o=.d)


.PHONY: serial clean rebuild