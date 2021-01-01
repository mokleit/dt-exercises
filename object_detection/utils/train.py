#!/usr/bin/env python3

import sys
sys.path.append('/content/drive/My Drive/Fall2020/IFT6757/Exercise3/dt-exercises/object_detection/exercise_ws/src/object_detection/include/object_detection')
import model

# MODEL_PATH="../exercise_ws/src/obj_det/include/model"
MODEL_PATH="/content/drive/My Drive/Fall2020/IFT6757/Exercise3/dt-exercises/object_detection/exercise_ws/src/obj_det/include/model"

def main():
    # TODO train loop here!
    # TODO don't forget to save the model's weights inside of f"{MODEL_PATH}/weights`!
    model.train()
    

if __name__ == "__main__":
    main()