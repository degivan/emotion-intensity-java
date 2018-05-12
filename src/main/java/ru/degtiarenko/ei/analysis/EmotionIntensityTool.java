package ru.degtiarenko.ei.analysis;

import org.jetbrains.annotations.NotNull;
import ru.degtiarenko.ei.analysis.tweets.AnalysedTweet;
import ru.degtiarenko.ei.analysis.tweets.Tweet;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;


public class EmotionIntensityTool {
    private static final String ANALYSE_COMMAND_REGEX = "analyse [\\w,\\s-,/]+\\.txt";
    private static final String ANY_SYMBOLS_REGEX = "[\\w,\\s-,/]*";
    private static final String EMOTION_FILENAME_FORMAT = ANY_SYMBOLS_REGEX + "%s" + ANY_SYMBOLS_REGEX + "\\." + "%s";
    private static final String MODEL_RESOLUTION = "h5";
    private static final String WORD_INDEX_RESOLUTION = "json";

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.out.println("Please, provide path to models folder.");
            System.exit(0);
        }
        EmotionIntensityAnalyzer analyzer = createAnalyzer(args[0]);
        Scanner scanner = new Scanner(System.in);

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
        File source = new File(tweetsFileName);
        Scanner scanner = new Scanner(source);

        while (scanner.hasNextLine()) {
            result.add(new Tweet(scanner.nextLine()));
        }

        scanner.close();
        return result;
    }

    @NotNull
    private static EmotionIntensityAnalyzer createAnalyzer(String pathToModels) throws Exception {
        String modelsDirectoryPath = getPathToModels(pathToModels);
        List<Path> filePaths = Files.list(Paths.get(modelsDirectoryPath)).collect(Collectors.toList());

        Map<Emotion, String> modelPaths = new HashMap<>();
        Map<Emotion, String> wordIndexPaths = new HashMap<>();
        for (Emotion emotion : Emotion.values()) {
            String emotionName = emotion.name().toLowerCase();

            String modelRegex = createEmotionPathPattern(emotionName, MODEL_RESOLUTION);
            String wordIndexRegex = createEmotionPathPattern(emotionName, WORD_INDEX_RESOLUTION);
            String pathToModel = findMatchingPath(filePaths, modelRegex);
            String pathToWordIndex = findMatchingPath(filePaths, wordIndexRegex);

            modelPaths.put(emotion, pathToModel);
            wordIndexPaths.put(emotion, pathToWordIndex);
        }

        return new EmotionIntensityAnalyzer(modelPaths, wordIndexPaths);
    }

    private static String findMatchingPath(List<Path> filePaths, String regex) throws FileNotFoundException {
        List<String> result = filePaths.stream()
                .map(path -> path.toAbsolutePath().toString())
                .filter(path -> path.matches(regex))
                .collect(Collectors.toList());
        if (result.isEmpty()) {
            throw new FileNotFoundException("Models folder should contain file matching " + regex + ".");
        } else if (result.size() > 1) {
            throw new IllegalArgumentException("Too many files matching " + regex + ".");
        }
        return result.get(0);
    }

    @NotNull
    private static String createEmotionPathPattern(String emotionName, String resolution) {
        return String.format(EMOTION_FILENAME_FORMAT, emotionName, resolution);
    }

    @NotNull
    private static String getPathToModels(String modelsFolderFileName) {
        String startingPath = Paths.get("").toAbsolutePath().toString();

        if (modelsFolderFileName.startsWith("/")) {
            modelsFolderFileName = modelsFolderFileName.substring(1);
            startingPath = "";
        }
        if (modelsFolderFileName.endsWith("/")) {
            modelsFolderFileName = modelsFolderFileName.substring(0, modelsFolderFileName.length() - 1);
        }

        StringBuilder builder = new StringBuilder();
        builder.append(startingPath)
                .append("/")
                .append(modelsFolderFileName)
                .append("/");
        return builder.toString();
    }

}
