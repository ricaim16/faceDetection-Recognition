import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class OpenCVFaceDetection {

    public static void main(String[] args) throws IOException {
        // Load OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Path to your input image
        String imagePath = "src/test/resources/largest_selfie.jpg";
        Path imageFile = Paths.get(imagePath);

        // Read image using OpenCV
        Mat image = Imgcodecs.imread(imageFile.toString());

        // Initialize face detector
        CascadeClassifier faceDetector = new CascadeClassifier();
        faceDetector.load("haarcascade_frontalface_default.xml"); // Load Haar Cascade classifier

        // Detect faces in the image
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);

        // Process each detected face
        int i = 0;
        for (Rect rect : faceDetections.toArray()) {
            // Draw rectangle around each face
            Imgproc.rectangle(image,
                    new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0), 3);

            // Crop the face from the image
            Mat croppedImage = new Mat(image, rect);

            // Save the cropped face image
            String outputDir = "build/output/faces";
            Files.createDirectories(Paths.get(outputDir));
            String outputPath = outputDir + "/face_" + i + ".png";
            Imgcodecs.imwrite(outputPath, croppedImage);

            System.out.println("Face " + i + " cropped and saved to: " + outputPath);
            i++;
        }

        // Save the annotated image with bounding boxes
        String annotatedImagePath = "build/output/annotated_image.png";
        Imgcodecs.imwrite(annotatedImagePath, image);

        System.out.println("Annotated image saved to: " + annotatedImagePath);
    }
}
