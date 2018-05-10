package ru.degtiarenko.ei.analysis;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
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
    private final static Map<Emotion, Integer> emotionPosition = new HashMap<>();

    private final Map<Emotion, MultiLayerNetwork> emotionNetworks = new HashMap<>();
    private final Map<Emotion, TweetTokenizer> tweetTokenizers = new HashMap<>();


    static {
        emotionPosition.put(Emotion.ANGER, 0);
        emotionPosition.put(Emotion.SADNESS, 1);
        emotionPosition.put(Emotion.JOY, 2);
        emotionPosition.put(Emotion.FEAR, 3);
    }

    public EmotionIntensityAnalyzer(Map<Emotion, String> modelPaths, Map<Emotion, String> pathsToWordIndex) throws Exception {
        for (Emotion emotion : modelPaths.keySet()) {
            String modelPath = modelPaths.get(emotion);
            MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights(modelPath, true);

            emotionNetworks.put(emotion, network);
        }
        for (Emotion emotion : pathsToWordIndex.keySet()) {
            String pathToWordIndex = pathsToWordIndex.get(emotion);
            tweetTokenizers.put(emotion, new TweetTokenizer(pathToWordIndex));
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
        List<Map<Emotion, Double>> result = new ArrayList<>();
        for (int i = 0; i < tweets.size(); i++) {
            result.add(new HashMap<>());
        }

        for (Emotion emotion : emotionNetworks.keySet()) {
            MultiLayerNetwork network = emotionNetworks.get(emotion);
            TweetTokenizer tweetTokenizer = tweetTokenizers.get(emotion);
            INDArray tweetX = tweetTokenizer.tokenize(tweets);
            INDArray res = network.output(tweetX);
            INDArray emotionRes = res.getColumn(emotionPosition.get(emotion));
            for (int i = 0; i < tweets.size(); i++) {
                double tweetRes = emotionRes.getDouble(i);
                result.get(i).put(emotion, tweetRes);
            }
        }

        return result;
    }
}
