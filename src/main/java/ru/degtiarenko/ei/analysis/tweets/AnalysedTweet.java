package ru.degtiarenko.ei.analysis.tweets;

import ru.degtiarenko.ei.analysis.Emotion;

import java.util.Map;

public class AnalysedTweet extends Tweet {
    private final Map<Emotion, Double> intensities;

    public AnalysedTweet(Map<Emotion, Double> intensities, Tweet tweet) {
        super(tweet);
        this.intensities = intensities;
    }

    public Map<Emotion, Double> getIntensities() {
        return intensities;
    }
}
