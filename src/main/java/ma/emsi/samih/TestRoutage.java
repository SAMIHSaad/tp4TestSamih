package ma.emsi.samih;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.service.AiServices;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class TestRoutage {

    private static final Logger log = LoggerFactory.getLogger(TestRoutage.class);

    public static void main(String[] args) {

        System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "debug");

        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY")) // Ensure GOOGLE_API_KEY is set as an environment variable
                .modelName("gemini-2.0-flash")
                .build();

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("text-embedding-004")
                .build();

        Path fineTuningRAGDocumentPath = Paths.get("src/main/resources/FineTuningRAG.pdf");
        Path restfullDocumentPath = Paths.get("src/main/resources/RESTfull.pdf");


        EmbeddingStore<TextSegment> fineTuningRAGEmbeddingStore = createAndIngestEmbeddingStore(fineTuningRAGDocumentPath, embeddingModel);
        EmbeddingStore<TextSegment> restfullEmbeddingStore = createAndIngestEmbeddingStore(restfullDocumentPath, embeddingModel);


        ContentRetriever fineTuningRAGContentRetriever = createContentRetriever(fineTuningRAGEmbeddingStore, embeddingModel);
        ContentRetriever restfullContentRetriever = createContentRetriever(restfullEmbeddingStore, embeddingModel);


        Map<ContentRetriever, String> retrieverDescriptions = new HashMap<>();
        retrieverDescriptions.put(fineTuningRAGContentRetriever, "Contains information about fine-tuning Retrieval Augmented Generation (RAG) models, large language models, and natural language processing.");
        retrieverDescriptions.put(restfullContentRetriever, "Contains information about RESTful APIs, web services, and software architecture.");

        QueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, retrieverDescriptions);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .build();


        System.out.println("--- Test avec une requête liée à l'IA ---");
        String aiQuery = "Qu'est-ce que le réglage fin de RAG ?";
        System.out.println("Utilisateur : " + aiQuery);
        System.out.println("Assistant : " + assistant.chat(aiQuery));

        System.out.println(" --- Test avec une requête liée aux API RESTful ---");
        String restfullQuery = "Expliquez les principes de REST.";
        System.out.println("Utilisateur : " + restfullQuery);
        System.out.println("Assistant : " + assistant.chat(restfullQuery));

        System.out.println(" --- Test avec une requête mixte (devrait idéalement en choisir une ou aucune, selon le LM) ---");
        String mixedQuery = "Comment l'IA peut-elle être utilisée avec les API RESTful ?";
        System.out.println("Utilisateur : " + mixedQuery);
        System.out.println("Assistant : " + assistant.chat(mixedQuery));
    }

    private static EmbeddingStore<TextSegment> createAndIngestEmbeddingStore(Path documentPath, EmbeddingModel embeddingModel) {
        log.info("Creating and ingesting embedding store for document: {}", documentPath);
        String extractedText;
        try (PDDocument pdfDocument = PDDocument.load(documentPath.toFile())) {
            PDFTextStripper pdfTextStripper = new PDFTextStripper();
            extractedText = pdfTextStripper.getText(pdfDocument);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load document: " + documentPath, e);
        }

        List<TextSegment> segments = new ArrayList<>();
        int segmentSize = 500;
        for (int i = 0; i < extractedText.length(); i += segmentSize) {
            int endIndex = Math.min(i + segmentSize, extractedText.length());
            segments.add(TextSegment.from(extractedText.substring(i, endIndex)));
        }

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        log.info("Ingested {} segments into embedding store for {}", segments.size(), documentPath.getFileName());
        return embeddingStore;
    }

    private static ContentRetriever createContentRetriever(EmbeddingStore<TextSegment> embeddingStore, EmbeddingModel embeddingModel) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.7)
                .build();
    }


}
