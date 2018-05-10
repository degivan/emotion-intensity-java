package ru.degtiarenko.ei.analysis;

import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
import org.nd4j.linalg.factory.Nd4j;
import ru.degtiarenko.ei.analysis.tweets.Tweet;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.util.Collections.singletonList;


public class Main {
    public static void main(String[] args) throws Exception {
        initMemory();

        String currDirectoryPath = Paths.get("").toAbsolutePath().toString();

        Map<Emotion, String> modelPaths = new HashMap<>();
        String pathToModel = currDirectoryPath + "/networks/best_model_joy.h5";
        modelPaths.put(Emotion.JOY, pathToModel);

        Map<Emotion, String> wordIndexPaths = new HashMap<>();
        String pathToWordIndex = currDirectoryPath + "/networks/word_index_joy.json";
        wordIndexPaths.put(Emotion.JOY, pathToWordIndex);

        EmotionIntensityAnalyzer analyzer = new EmotionIntensityAnalyzer(modelPaths, wordIndexPaths);

        final List<Tweet> testList = singletonList(new Tweet("I am angry"));
        System.out.println(analyzer.analyseData(testList));
    }

    private static void initMemory() {
        WorkspaceConfiguration mmap = WorkspaceConfiguration.builder()
                .initialSize(100000000L)
                .policyLocation(LocationPolicy.MMAP)
                .build();
        Nd4j.getWorkspaceManager().getAndActivateWorkspace(mmap, "M2"); }
}
