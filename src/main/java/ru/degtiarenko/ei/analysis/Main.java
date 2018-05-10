package ru.degtiarenko.ei.analysis;

import ru.degtiarenko.ei.analysis.tweets.Tweet;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.util.Collections.singletonList;


public class Main {
    public static void main(String[] args) throws Exception {
        String currDirectoryPath = Paths.get("").toAbsolutePath().toString();

        Map<Emotion, String> modelPaths = new HashMap<>();
        String pathToJoyModel = currDirectoryPath + "/networks/best_model_joy.h5";
        String pathToSadnessModel = currDirectoryPath + "/networks/best_model_sadness.h5";
        modelPaths.put(Emotion.JOY, pathToJoyModel);
        modelPaths.put(Emotion.SADNESS, pathToSadnessModel);

        Map<Emotion, String> wordIndexPaths = new HashMap<>();
        String pathToJoyWordIndex = currDirectoryPath + "/networks/word_index_joy.json";
        String pathToSadnessWordIndex = currDirectoryPath + "/networks/word_index_sadness.json";
        wordIndexPaths.put(Emotion.JOY, pathToJoyWordIndex);
        wordIndexPaths.put(Emotion.SADNESS, pathToSadnessWordIndex);

        EmotionIntensityAnalyzer analyzer = new EmotionIntensityAnalyzer(modelPaths, wordIndexPaths);

        final List<Tweet> testList = singletonList(new Tweet(""));
        System.out.println(analyzer.analyseData(testList));
    }

}
