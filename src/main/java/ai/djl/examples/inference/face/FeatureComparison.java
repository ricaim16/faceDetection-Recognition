

package ai.djl.examples.inference.face;

import ai.djl.ModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class FeatureComparison {

    private static final Logger logger = LoggerFactory.getLogger(FeatureComparison.class);

    private FeatureComparison() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Path imageFile1 = Paths.get("src/test/resources/kana1.jpg");
        Image img1 = ImageFactory.getInstance().fromFile(imageFile1);

        Path imageFile2 = Paths.get("src/test/resources/kana2.jpg");
        Image img2 = ImageFactory.getInstance().fromFile(imageFile2);

        RetinaFaceDetection rt = new RetinaFaceDetection();

        DetectedObjects detectedObjects = rt.getFace(img1);
        DetectedObjects.DetectedObject box = (DetectedObjects.DetectedObject)detectedObjects.items().get(0);
        Image face1 = cropFace(img1, box);

        DetectedObjects detectedObjects2 = rt.getFace(img2);
        DetectedObjects.DetectedObject box2 = (DetectedObjects.DetectedObject)detectedObjects2.items().get(0);
        Image face2 = cropFace(img2, box2);




        float[] feature1 = FeatureExtraction.predict(face1);

        float[] feature2 = FeatureExtraction.predict(face2);

        logger.info(Float.toString(calculateSimilarity(feature1, feature2)) + "%");
    }

    public static Image cropFace(Image img1, DetectedObjects.DetectedObject box){
        BoundingBox boundingBox = box.getBoundingBox();
        Rectangle rec = boundingBox.getBounds();
        int x = (int)(rec.getX() * img1.getWidth());
        int y = (int)(rec.getY() * img1.getHeight());
        int w = (int)(rec.getWidth() * img1.getWidth());
        int h = (int)(rec.getHeight() * img1.getHeight());
        return img1.getSubImage(x, y, w, h);
    }
    public static float calculateSimilarity(float[] feature1, float[] feature2) {
        float ret = 0.0f;
        float mod1 = 0.0f;
        float mod2 = 0.0f;
        int length = feature1.length;
        for (int i = 0; i < length; ++i) {
            ret += feature1[i] * feature2[i];
            mod1 += feature1[i] * feature1[i];
            mod2 += feature2[i] * feature2[i];
        }
        float similarity = ret / (float) (Math.sqrt(mod1) * Math.sqrt(mod2));
        float similarityPercent = (similarity + 1) * 50; // Convert similarity to a percentage
        return similarityPercent;
    }
}