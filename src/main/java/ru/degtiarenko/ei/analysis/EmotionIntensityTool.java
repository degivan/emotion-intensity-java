package ru.degtiarenko.ei.analysis;

import org.jetbrains.annotations.NotNull;
import ru.degtiarenko.ei.analysis.tweets.AnalysedTweet;
import ru.degtiarenko.ei.analysis.tweets.Tweet;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Paths;
import java.util.*;


public class EmotionIntensityTool {
    private static final String MODELS_FOLDER_FILENAME = "/networks/";
    private static final String MODEL_FILENAME_FORMAT = "best_model_%s.h5";
    private static final String WORD_INDEX_FILENAME_FORMAT = "word_index_%s.json";
    private static final String ANALYSE_COMMAND_REGEX = "analyse [\\w,\\s-,/]+\\.txt";

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);
        EmotionIntensityAnalyzer analyzer = createAnalyzer();

        while (true) {
            String command = scanner.nextLine();
            if (command.equals("exit")) {
                scanner.close();
                System.exit(0);
            } else if (command.matches(ANALYSE_COMMAND_REGEX)) {
                try {
                    String tweetsFileName = command.split(" ")[1];
                    List<Tweet> tweets = extractTweets(tweetsFileName);
                    List<AnalysedTweet> analysedTweets = analyzer.analyseData(tweets);
                    System.out.println(analysedTweets);
                } catch (FileNotFoundException ex) {
                    System.out.println("File not found.");
                }
            } else {
                System.out.println("Unknown command.");
            }
        }
    }

    private static List<Tweet> extractTweets(String tweetsFileName) throws FileNotFoundException {
        List<Tweet> result = new ArrayList<>();
        Scanner scanner = new Scanner(new File(tweetsFileName));

        while(scanner.hasNextLine()) {
            result.add(new Tweet(scanner.nextLine()));
        }

        scanner.close();
        return result;
    }

    @NotNull
    private static EmotionIntensityAnalyzer createAnalyzer() throws Exception {
        String modelsDirectoryPath = Paths.get("").toAbsolutePath().toString() + MODELS_FOLDER_FILENAME;

        Map<Emotion, String> modelPaths = new HashMap<>();
        Map<Emotion, String> wordIndexPaths = new HashMap<>();
        for (Emotion emotion: Emotion.values()) {
            String emotionName = emotion.name().toLowerCase();

            String modelFileName = String.format(MODEL_FILENAME_FORMAT, emotionName);
            String wordIndexFileName = String.format(WORD_INDEX_FILENAME_FORMAT, emotionName);
            String pathToModel = modelsDirectoryPath + modelFileName;
            String pathToWordIndex = modelsDirectoryPath + wordIndexFileName;

            modelPaths.put(emotion, pathToModel);
            wordIndexPaths.put(emotion,pathToWordIndex);
        }

        return new EmotionIntensityAnalyzer(modelPaths, wordIndexPaths);
    }

}
