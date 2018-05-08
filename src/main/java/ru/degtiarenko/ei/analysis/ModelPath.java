package ru.degtiarenko.ei.analysis;

public class ModelPath {
    private final String pathToJson;
    private final String pathToWeights;

    public ModelPath(String pathToJson, String pathToWeights) {
        this.pathToJson = pathToJson;
        this.pathToWeights = pathToWeights;
    }

    public String getPathToWeights() {
        return pathToWeights;
    }

    public String getPathToJson() {
        return pathToJson;
    }
}
