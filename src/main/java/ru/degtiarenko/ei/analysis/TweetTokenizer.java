package ru.degtiarenko.ei.analysis;

import com.google.common.collect.Lists;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import ru.degtiarenko.ei.analysis.tweets.Tweet;

import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class TweetTokenizer {
    private final static int MAX_WORDS_COUNT = 52; //TODO: extract from some source
    private final Map<String, Double> wordIndexMap;
    private final TokenizerFactory tokenizerFactory;

    public TweetTokenizer(String pathToWordIndex) throws IOException, ParseException {
        this.wordIndexMap = extractWordIndex(pathToWordIndex);
        this.tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }

    private static Map<String, Double> extractWordIndex(String pathToWordIndex) throws IOException, ParseException {
        Map<String, Double> result = new HashMap<>();
        JSONParser parser = new JSONParser();
        JSONObject jsonWordIndex = (JSONObject) parser.parse(new FileReader(pathToWordIndex));
        for (Object word : jsonWordIndex.keySet()) {
            Double index = Double.valueOf((Long) jsonWordIndex.get(word));
            result.put(String.valueOf(word), index);
        }
        return result;
    }

    //TODO
    public INDArray tokenize(List<Tweet> tweets) {
        double[][][] featuresMatrix = new double[tweets.size()][1][MAX_WORDS_COUNT];
        for (double[][] featuresRow : featuresMatrix) {
            Arrays.fill(featuresRow[0], 0.0);
        }
        for (int i = 0; i < tweets.size(); i++) {
            Tweet tweet = tweets.get(i);
            List<Double> indexes = tweetToIndexes(tweet);

            for (int j = 0; j < indexes.size(); j++) {
                featuresMatrix[i][0][MAX_WORDS_COUNT - indexes.size() + j] = indexes.get(j);
            }
        }

        List<INDArray> ndArrays = Arrays.stream(featuresMatrix)
                .map(Nd4j::create)
                .collect(Collectors.toList());
        return Nd4j.create(ndArrays, new int[]{tweets.size(), 1, MAX_WORDS_COUNT});
    }

    private List<Double> tweetToIndexes(Tweet tweet) {
        List<String> tokens = tokenizerFactory.create(tweet.getText()).getTokens();
        return tokens.stream()
                .map(wordIndexMap::get)
                .collect(Collectors.toList());
    }
}
