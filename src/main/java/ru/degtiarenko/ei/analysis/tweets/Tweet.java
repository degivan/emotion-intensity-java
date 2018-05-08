package ru.degtiarenko.ei.analysis.tweets;

public class Tweet {
    private final String text;

    public Tweet(String text) {
        this.text = text;
    }

    public Tweet(Tweet tweet) {
        this(tweet.getText());
    }

    public String getText() {
        return text;
    }
}
