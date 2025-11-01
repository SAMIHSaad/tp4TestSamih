package ma.emsi.samih;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class TestRoutage {

    private static final Logger log = LoggerFactory.getLogger(TestRoutage.class);

    public static void main(String[] args) {

        System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "debug");

        // Verify environment variables
        String geminiKey = System.getenv("GEMINI_KEY");
        String tavilyKey = System.getenv("TAVILY_KEY");


        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-2.0-flash")
                .build();

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(geminiKey)
                .modelName("text-embedding-004")
                .build();

        Path fineTuningRAGDocumentPath = Paths.get("src/main/resources/FineTuningRAG.pdf");

        EmbeddingStore<TextSegment> fineTuningRAGEmbeddingStore = createAndIngestEmbeddingStore(
                fineTuningRAGDocumentPath, embeddingModel);

        ContentRetriever fineTuningRAGContentRetriever = createContentRetriever(
                fineTuningRAGEmbeddingStore, embeddingModel);

        TavilyWebSearchEngine tavilyWebSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever webSearchContentRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(tavilyWebSearchEngine)
                .build();

        QueryRouter queryRouter = new DefaultQueryRouter(
                fineTuningRAGContentRetriever,
                webSearchContentRetriever
        );

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("--- Test avec 'Bonjour' (ne devrait pas utiliser le RAG) ---");
        String helloQuery = "Bonjour.";
        System.out.println("Utilisateur : " + helloQuery);
        System.out.println("Assistant : " + assistant.chat(helloQuery));

        System.out.println("\n--- Test avec une requête liée à l'IA (devrait utiliser le RAG) ---");
        String aiQuery = "Qu'est-ce que le réglage fin de RAG ?";
        System.out.println("Utilisateur : " + aiQuery);
        System.out.println("Assistant : " + assistant.chat(aiQuery));

        System.out.println("\n--- Test avec une requête non liée à l'IA (ne devrait pas utiliser le RAG) ---");
        String nonAiQuery = "Quelle est la capitale de la France ?";
        System.out.println("Utilisateur : " + nonAiQuery);
        System.out.println("Assistant : " + assistant.chat(nonAiQuery));
    }

    private static EmbeddingStore<TextSegment> createAndIngestEmbeddingStore(
            Path documentPath, EmbeddingModel embeddingModel) {

        log.info("Création et ingestion du magasin d'embeddings pour le document : {}", documentPath);

        String extractedText;
        try (PDDocument pdfDocument = PDDocument.load(documentPath.toFile())) {
            PDFTextStripper pdfTextStripper = new PDFTextStripper();
            extractedText = pdfTextStripper.getText(pdfDocument);
        } catch (IOException e) {
            throw new RuntimeException("Échec du chargement du document : " + documentPath, e);
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

        log.info("Ingestion de {} segments dans le magasin d'embeddings pour {}",
                segments.size(), documentPath.getFileName());

        return embeddingStore;
    }

    private static ContentRetriever createContentRetriever(
            EmbeddingStore<TextSegment> embeddingStore, EmbeddingModel embeddingModel) {

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.7)
                .build();
    }
}
