# Emotion Intensity Tool
Java import of emotion intensity prediction model.

## Prerequisites

* Java 8 (or more)
* Gradle 3.0 (or more)
* Folder with models and word-index files for all emotions (name of every file should contain corresponding emotion)

## Usage

* Create jar using `gradle shadowJar` command.
* Start using `java -jar <jarName> <pathToFolderWithModels>`
* Type `analyse <filename.txt` to get emotion intensities for all texts (each on separate line)
* Type `exit` to stop executing


## License

GNU General Public License v3.0

See [LICENSE](../master/LICENSE) to see the full text.
