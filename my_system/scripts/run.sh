#!/bin/bash

# Setup SDK library paths for runtime
export LD_LIBRARY_PATH=$BASE_DIR/$EXPORT_LIB_M1_SDK_ROOT_PATH/lib:$BASE_DIR/$EXPORT_LIB_M1_SDK_ROOT_PATH/third_party/lib:$LD_LIBRARY_PATH

chmod +x ./my_system
./my_system
