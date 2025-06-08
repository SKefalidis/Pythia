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

public class OntologyExtractor {
    static Map<String, HashSet<String>> entityToClassMap = new ConcurrentHashMap<>();
    static Set<String> classes = new HashSet<>();
    static ArrayList<String> classPrefixes = new ArrayList<>();

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
    
    static class myTriple {
        String subject;
        String predicate;
        String object;

        myTriple(String subject, String predicate, String object) {
            this.subject = subject;
            this.predicate = predicate;
            this.object = object;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof myTriple)) return false;
            myTriple other = (myTriple) o;
            return Objects.equals(subject, other.subject) &&
                   Objects.equals(predicate, other.predicate) &&
                   Objects.equals(object, other.object);
        }
    
        @Override
        public int hashCode() {
            return Objects.hash(subject, predicate, object);
        }
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
        options.addOption("t", "threads", true, "Number of threads");
        options.addOption("p", "prefixes", true, "Possible prefixes of classes");

        Path inputFile = null;
        Path outputFile = null;
        int numThreads = -1;

        try {
            // Parse the command-line arguments
            CommandLine cmd = parser.parse(options, args);

            // Get input and output paths
            inputFile = Paths.get(cmd.getOptionValue("i"));
            outputFile = Paths.get(cmd.getOptionValue("o"));
            numThreads = cmd.hasOption("t") ? Integer.parseInt(cmd.getOptionValue("t")) : Runtime.getRuntime().availableProcessors();
            classPrefixes = cmd.hasOption("p") ? parseList(cmd.getOptionValue("p")) : new ArrayList<>();

            // Output parsed values
            System.out.println("Input File: " + inputFile);
            System.out.println("Output File: " + outputFile);
            System.out.println("Number of Threads: " + numThreads);
            System.out.println("Class Prefixes: " + classPrefixes);
        } catch (Exception e) {
            System.err.println("Error parsing arguments: " + e.getMessage());
            new HelpFormatter().printHelp("EntityExtractor", options);
            System.exit(1);
        }

        // Extract entities to class map (Preprocessing)
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<Map<String, HashSet<String>>> completionService = 
            new ExecutorCompletionService<>(executor);

        AtomicInteger fileCount = new AtomicInteger(0);
        if (Files.isDirectory(inputFile)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(inputFile)) {
                for (Path file : stream) {
                    if (Files.isRegularFile(file)) {
                        fileCount.incrementAndGet();
                        completionService.submit(() -> getEntitiesToClassMap(file));
                    }
                }
            }
        } else if (Files.isRegularFile(inputFile)) {
            // If input is a file, process it directly
            final var inputFileTemp = inputFile; // just to bypass  the need for final
            fileCount.incrementAndGet();
            completionService.submit(() -> getEntitiesToClassMap(inputFileTemp));
        } else {
            System.err.println("Input must be a directory or a file.");
            System.exit(1);
        }

        for (int i = 0; i < fileCount.get(); i++) {
            try {
                Map<String, HashSet<String>> results = completionService.take().get();
                entityToClassMap.putAll(results);
            } catch (ExecutionException e) {
                System.err.println("Error processing file: " + e.getCause().getMessage());
            }
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        System.out.println("Found " + classes.size() + " classes");
        System.out.println("Found " + entityToClassMap.size() + " entity-class mappings");

        // Clear output file
        Files.deleteIfExists(outputFile);
        Files.createFile(outputFile);

        // Replace entities with classes
        executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<Set<myTriple>> triplesCompletionService = 
            new ExecutorCompletionService<>(executor);

        fileCount = new AtomicInteger(0);
        if (Files.isDirectory(inputFile)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(inputFile)) {
                for (Path file : stream) {
                    if (Files.isRegularFile(file)) {
                        fileCount.incrementAndGet();
                        triplesCompletionService.submit(() -> processFile(file));
                    }
                }
            }
        } else if (Files.isRegularFile(inputFile)) {
            // If input is a file, process it directly
            final var inputFileTemp = inputFile; // just to bypass  the need for final
            fileCount.incrementAndGet();
            triplesCompletionService.submit(() -> processFile(inputFileTemp));
        } else {
            System.err.println("Input must be a directory or a file.");
            System.exit(1);
        }

        Set<myTriple> uniqueTriples = new LinkedHashSet<>(); // Preserves insertion order
        try {
            for (int i = 0; i < fileCount.get(); i++) {
                try {
                    Set<myTriple> results = triplesCompletionService.take().get();
                    uniqueTriples.addAll(results); // Duplicates automatically removed
                } catch (ExecutionException e) {
                    System.err.println("Error processing file: " + e.getCause().getMessage());
                    e.getCause().printStackTrace();
                }
            }

            // Now write only unique triples
            try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
                for (myTriple data : uniqueTriples) {
                    writer.write("<" + data.subject + "> <" + data.predicate + "> <" + data.object + "> .\n");
                }
            }
        } finally {
            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.HOURS);
        }

        System.out.println("Output written to " + outputFile);
        System.out.println("Finished processing files.");
    }

    private static Map<String, HashSet<String>> getEntitiesToClassMap(Path file) throws IOException {        
        Map<String, HashSet<String>> entitiesToClassMap = new ConcurrentHashMap<>();

        StreamRDF processor = new StreamRDF() {
            @Override
            public void triple(Triple triple) {
                processTypeTriple(triple, entitiesToClassMap);
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

        return entitiesToClassMap;
    }

    private static void processTypeTriple(Triple triple, Map<String, HashSet<String>> entitiesToClassMap) {
        if (!triple.getSubject().isURI()) 
            return; // Skip blank nodes
        if (!triple.getObject().isURI())
            return; // Skip literals

        String subject = triple.getSubject().getURI();
        String predicate = triple.getPredicate().getURI();
        String object = triple.getObject().getURI();

        if (predicate.contains("http://www.w3.org/1999/02/22-rdf-syntax-ns#type") || predicate.contains("http://www.wikidata.org/prop/direct/P31")) {
            if (classPrefixes.size() == 0 || classPrefixes.stream().anyMatch(object::contains)) {
                if (!entitiesToClassMap.containsKey(subject)) {
                    entitiesToClassMap.put(subject, new HashSet<>());
                }
                entitiesToClassMap.get(subject).add(object);
                classes.add(object);
            }
        }
    }

    private static Set<myTriple> processFile(Path file) throws IOException {        
        Set<myTriple> triples = new HashSet<>();

        StreamRDF processor = new StreamRDF() {
            @Override
            public void triple(Triple triple) {
                processTriple(triple, triples);
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

        return triples;
    }

    private static void processTriple(Triple triple, Set<myTriple> triples) {
        if (!triple.getSubject().isURI()) 
            return; // Skip blank nodes
        if (!triple.getObject().isURI())
            return; // Skip literals

        String subject = triple.getSubject().getURI();
        String predicate = triple.getPredicate().getURI();
        String object = triple.getObject().getURI();

        if (entityToClassMap.containsKey(subject) && entityToClassMap.containsKey(object)) {
            // If both subject and object are entities, replace them with their classes
            for (String subjectClass : entityToClassMap.get(subject)) {
                for (String objectClass : entityToClassMap.get(object)) {
                    triples.add(new myTriple(subjectClass, predicate, objectClass));
                }
            }
        }
    }

}