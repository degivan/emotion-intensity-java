package ru.degtiarenko.ei.analysis;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import ru.degtiarenko.ei.analysis.tweets.AnalysedTweet;
import ru.degtiarenko.ei.analysis.tweets.Tweet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Client wraper for analyzer server.
 */
public class EmotionIntensityAnalyzer {
    private final Map<Emotion, MultiLayerNetwork> emotionNetworks = new HashMap<>();

    public EmotionIntensityAnalyzer(Map<Emotion, String> modelPaths) throws Exception {
        for (Emotion emotion : modelPaths.keySet()) {
            String modelPath = modelPaths.get(emotion);
            MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights(modelPath, true);
            System.out.println(network.summary());
            emotionNetworks.put(emotion, network);
        }
    }

    public List<AnalysedTweet> analyseData(List<Tweet> tweets) throws IOException {
        List<Map<Emotion, Double>> intensities = countIntensity(tweets);

        List<AnalysedTweet> result = new ArrayList<>();
        for (int i = 0; i < Math.max(intensities.size(), tweets.size()); i++) {
            Tweet tweet = tweets.get(i);
            Map<Emotion, Double> intensity = intensities.get(i);
            result.add(new AnalysedTweet(intensity, tweet));
        }
        return result;
    }

    private List<Map<Emotion, Double>> countIntensity(List<Tweet> tweets) throws IOException {
        //TODO:
        return null;
    }
}
