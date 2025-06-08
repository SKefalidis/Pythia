import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.jena.graph.Triple;
import org.apache.jena.riot.Lang;
import org.apache.jena.riot.RDFParser;
import org.apache.jena.riot.system.StreamRDF;
import org.apache.jena.sparql.core.Quad;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;


// FIXME: Misses some predicates. Check with beastiary, it is apparent when you go through the file.
// FIXME: If there are no labels, maybe we should create a label from the URI? Split the last part of the URI with spaces? But this could mess up IDs.
public class EntityExtractor {
    static ArrayList<String> label_predicates = new ArrayList<>();
    static ArrayList<String> description_predicates = new ArrayList<>();
    static ArrayList<String> entity_prefixes = new ArrayList<>();
    static boolean filter_entities = false;
    
    static class UriData {
        final String uri;
        ArrayList<String> labels = new ArrayList<>();
        String description = "";
        String fallbackLabel = "";
        String fallbackDescription = "";
        boolean isClass = false;
        boolean isPredicate = false;
        boolean satisfiesPrefixes = false;

        UriData(String uri) {
            this.uri = uri;
        }

        void finalizeValues() {
            if (labels.isEmpty() && !fallbackLabel.isEmpty()) {
                labels.add(fallbackLabel);
            }
            if (description.isEmpty() && !fallbackDescription.isEmpty()) {
                description = fallbackDescription;
            }
        }
    }

    private static ArrayList<String> parseList(String listStr) {
        var list = new ArrayList<String>();
        if (listStr != null && !listStr.isEmpty()) {
            // Split the comma-separated list into items and add them to the list
            String[] items = listStr.split(",");
            for (String item : items) {
                list.add(item.trim()); // Trim any leading or trailing spaces
            }
        }
        return list;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: EntityExtractor <inputDir> <outputFile> <LABEL_URI_1,LABEL_URI_2...> <DESC_URI_1,DESC_URI_2...> [--filter] [numThreads]");
            System.exit(1);
        }

        // Create the command-line parser
        CommandLineParser parser = new DefaultParser();
        Options options = new Options();

        // Define options for the arguments
        options.addRequiredOption("i", "input", true, "Input directory/file");
        options.addRequiredOption("o", "output", true, "Output file");
        options.addOption("l", "labels", true, "Label URIs");
        options.addOption("d", "descriptions", true, "Description URIs");
        options.addOption("t", "threads", true, "Number of threads");
        options.addOption("f", "filter", false, "Filter entities without labels");
        options.addOption("p", "prefixes", true, "Possible prefixes of entities");
        
        Path inputFile = null;
        Path outputFile = null;
        int numThreads = -1;

        try {
            // Parse the command-line arguments
            CommandLine cmd = parser.parse(options, args);

            // Get input and output paths
            inputFile = Paths.get(cmd.getOptionValue("i"));
            outputFile = Paths.get(cmd.getOptionValue("o"));

            label_predicates = cmd.hasOption("l") ? parseList(cmd.getOptionValue("l")) : new ArrayList<>();
            description_predicates = cmd.hasOption("d") ? parseList(cmd.getOptionValue("d")) : new ArrayList<>();
            entity_prefixes = cmd.hasOption("p") ? parseList(cmd.getOptionValue("p")) : new ArrayList<>();

            numThreads = cmd.hasOption("t") ? Integer.parseInt(cmd.getOptionValue("t")) : Runtime.getRuntime().availableProcessors();
            filter_entities = cmd.hasOption("f") ? true : false;

            // Output parsed values
            System.out.println("Input File: " + inputFile);
            System.out.println("Output File: " + outputFile);
            System.out.println("Number of Threads: " + numThreads);
            System.out.println("Labels: " + label_predicates);
            System.out.println("Descriptions: " + description_predicates);
            System.out.println("Entity Prefixes: " + entity_prefixes);
        } catch (Exception e) {
            System.err.println("Error parsing arguments: " + e.getMessage());
            new HelpFormatter().printHelp("EntityExtractor", options);
            System.exit(1);
        }

        // Clear output file
        Files.deleteIfExists(outputFile);
        Files.createFile(outputFile);

        // Process files in parallel
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<List<UriData>> completionService = 
            new ExecutorCompletionService<>(executor);

        AtomicInteger fileCount = new AtomicInteger(0);
        if (Files.isDirectory(inputFile)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(inputFile)) {
                for (Path file : stream) {
                    if (Files.isRegularFile(file)) {
                        fileCount.incrementAndGet();
                        completionService.submit(() -> processFile(file));
                    }
                }
            }
        } else if (Files.isRegularFile(inputFile)) {
            // If input is a file, process it directly
            final var inputFileTemp = inputFile; // just to bypass  the need for final
            fileCount.incrementAndGet();
            completionService.submit(() -> processFile(inputFileTemp));
        } else {
            System.err.println("Input must be a directory or a file.");
            System.exit(1);
        }

        // Write results with CSV header
        try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            // Process results
            for (int i = 0; i < fileCount.get(); i++) {
                try {
                    List<UriData> results = completionService.take().get();
                    // System.out.println("Processed file " + (i + 1) + " of " + fileCount.get());
                    // System.out.println("Found " + results.size() + " subjects.");
                    for (UriData data : results) {
                        if (data.isClass) continue; // Skip classes
                        if (data.isPredicate) continue; // Skip predicates
                        if (data.satisfiesPrefixes == false && entity_prefixes.size() > 0) continue; // Skip entities that don't satisfy prefixes
                        if (filter_entities && data.labels.isEmpty()) continue; // Skip entities without labels
                        writer.write(data.uri);
                        if (data.labels.isEmpty() == false)
                            for (var l : data.labels)
                                writer.write(", " + l.replace('\n', ' '));
                        if (data.description.isEmpty() == false)
                            writer.write(", " + data.description.replace('\n', ' '));
                        writer.newLine();
                    }
                } catch (ExecutionException e) {
                    System.err.println("Error processing file: " + e.getCause().getMessage());
                }
            }
        } finally {
            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.HOURS);
        }
    }

    private static List<UriData> processFile(Path file) throws IOException {
        Map<String, UriData> uris = new ConcurrentHashMap<>();
        
        StreamRDF processor = new StreamRDF() {
            @Override
            public void triple(Triple triple) {
                processTriple(triple, uris);
            }

            @Override public void start() {}
            @Override public void quad(Quad quad) {}
            @Override public void base(String base) {}
            @Override public void prefix(String prefix, String iri) {}
            @Override public void finish() {}
        };

        RDFParser.source(file.toString())
            .lang(Lang.NTRIPLES)
            .parse(processor);

        if (filter_entities) {
            return uris.values().stream()
                .peek(UriData::finalizeValues)
                .filter(data -> !data.labels.isEmpty())
                .collect(Collectors.toList());
        } else {
            return uris.values().stream()
                .peek(UriData::finalizeValues)
                .collect(Collectors.toList());
        }
    }

    private static void processTriple(Triple triple, Map<String, UriData> uris) {
        String subject;
        if (!triple.getSubject().isURI()) 
            return; // Skip blank nodes
        subject = triple.getSubject().getURI();
        uris.computeIfAbsent(subject, UriData::new);
        if (entity_prefixes.size() > 0 && entity_prefixes.stream().anyMatch(subject::contains)) {
            uris.get(subject).satisfiesPrefixes = true;
        }

        String predicate = triple.getPredicate().getURI();
        uris.computeIfAbsent(predicate, UriData::new).isPredicate = true;
        
        String object = "";
        if (triple.getObject().isURI()) {
            object = triple.getObject().getURI();
            uris.computeIfAbsent(object, UriData::new);
            if (predicate.contains("http://www.w3.org/1999/02/22-rdf-syntax-ns#type") || predicate.contains("http://www.wikidata.org/prop/direct/P31")) {
                uris.get(object).isClass = true;
                return;
            } else if (entity_prefixes.size() > 0 && entity_prefixes.stream().anyMatch(object::contains)) {
                uris.get(object).satisfiesPrefixes = true;
            }
        } else {
            object = triple.getObject().isLiteral() ?
                    triple.getObject().getLiteralLexicalForm() :
                    triple.getObject().toString();
        }
        
        if (filter_entities && !label_predicates.contains(predicate) && !description_predicates.contains(predicate)) {
            return;
        }
        
        String lang = triple.getObject().isLiteral() ? triple.getObject().getLiteralLanguage() : "";
        UriData data = uris.get(subject);
        boolean isEnglish = lang.isEmpty() || lang.equals("en");

        if (label_predicates.contains(predicate)) {
            if (isEnglish && object.length() > 1) {
                data.labels.add(object);
            } else if (!isEnglish && data.fallbackLabel.isEmpty()) {
                data.fallbackLabel = object;
            }
        } else if (description_predicates.contains(predicate)) {
            if (isEnglish && data.description.isEmpty() && object.length() > 1) {
                data.description = object;
            } else if (!isEnglish && data.fallbackDescription.isEmpty()) {
                data.fallbackDescription = object;
            }
        }
    }

}