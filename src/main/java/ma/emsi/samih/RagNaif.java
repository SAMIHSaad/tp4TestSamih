package ma.emsi.samih;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;


import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static java.time.Duration.ofSeconds;

public class RagNaif {
    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
    public static void main(String[] args) throws IOException, URISyntaxException {


        Path documentPath = Paths.get(RagNaif.class.getResource("/FineTuningRAG.pdf").toURI());

        DocumentParser documentParser = new ApacheTikaDocumentParser();

        Document document = loadDocument(documentPath, documentParser);


        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        System.out.println("Phase 1: Embeddings enregistrés.");


        String llmKey = System.getenv("GEMINI_KEY");


        ChatModel model = GoogleAiGeminiChatModel
                .builder()
                .apiKey(llmKey)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .modelName("gemini-2.5-flash") // Nom du modèle que vous utilisez
                .build();

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10)) // Mémoire pour 10 messages
                .contentRetriever(contentRetriever)
                .build();

        System.out.println("Phase 2: Assistant prêt. Vous pouvez poser vos questions (tapez 'quitter' pour arrêter).");

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("\nVotre question: ");
            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("quitter")) {
                break;
            }

            String response = assistant.chat(question);
            System.out.println("Réponse de l'assistant: " + response);
        }

        scanner.close();
        System.out.println("Fin de la session.");
    }
}