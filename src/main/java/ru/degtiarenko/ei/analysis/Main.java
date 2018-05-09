package ru.degtiarenko.ei.analysis;

import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;


public class Main {
    public static void main(String[] args) throws Exception {
        initMemory();
        Map<Emotion, String> modelPaths = new HashMap<>();
        String currDirectoryPath = Paths.get("").toAbsolutePath().toString();
        String pathToModel = currDirectoryPath + "/networks/best_model_joy.h5";

        modelPaths.put(Emotion.JOY, pathToModel);

        EmotionIntensityAnalyzer analyzer = new EmotionIntensityAnalyzer(modelPaths);
    }

    private static void initMemory() {
        WorkspaceConfiguration mmap = WorkspaceConfiguration.builder()
                .initialSize(100000000L)
                .policyLocation(LocationPolicy.MMAP)
                .build();
        Nd4j.getWorkspaceManager().getAndActivateWorkspace(mmap, "M2"); }
}
