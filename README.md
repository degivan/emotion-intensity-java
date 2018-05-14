# Emotion Intensity Tool
Java import of emotion intensity prediction model.

## Prerequisites

* Java 8 (or more)
* Gradle 3.0 (or more)
* CUDA 8.0 or 9.0 (optional)

## Usage

#### Train models
* Put in one folder with script:
    * dirty_data -- subfolder `labeled` should contain unaccurate data you want to train with
    * train_data -- train data should be in files with name `EI-reg-en_<emotion>_train.txt`
    * twitter_sgns_subset.txt.gz -- pre-trained word vectors

* Run `build_emotion_model.py <emotion>` for each emotion (anger, fear, joy and sadness)
* Models and word-indexes are in folder `networks`
* Run `change_rec_init.py <pathToFolderWithModels>` to avoid enormous memory usage in next steps

#### Use models
* Create jar using `gradle shadowJar` command. If you want to use GPU instead of CPU, you can build using `gradle shadowJar -Pbackend=GPU-CUDA-8.0` or `gradle shadowJar -Pbackend=GPU-CUDA-9.0` depending on your version of CUDA.
* Start using `java -jar <jarName> <pathToFolderWithModels>`. Consider these VM options: `-Xms1G -Xmx2G -Dorg.bytedeco.javacpp.maxbytes=4G -Dorg.bytedeco.javacpp.maxphysicalbytes=4G`. This should provide enough memory to application to run.
* Type `analyse <filename.txt` to get emotion intensities for all texts (each on separate line)
* Type `exit` to stop executing


## License

GNU General Public License v3.0

See [LICENSE](../master/LICENSE) to see the full text.
