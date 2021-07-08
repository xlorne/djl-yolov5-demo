package com.codingapi.djldemo;


import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.paddlepaddle.zoo.cv.imageclassification.PpWordRotateTranslator;
import ai.djl.paddlepaddle.zoo.cv.objectdetection.PpWordDetectionTranslator;
import ai.djl.paddlepaddle.zoo.cv.wordrecognition.PpWordRecognitionTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author lorne
 * @since 1.0.0
 */
public class PaddleOCR {

    static Image  getSubImage(Image img, BoundingBox box) {
        Rectangle rect = box.getBounds();
        double[] extended = extendRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight());
        int width = img.getWidth();
        int height = img.getHeight();
        int[] recovered = {
                (int) (extended[0] * width),
                (int) (extended[1] * height),
                (int) (extended[2] * width),
                (int) (extended[3] * height)
        };
        return img.getSubimage(recovered[0], recovered[1], recovered[2], recovered[3]);
    }

    static  double[] extendRect(double xmin, double ymin, double width, double height) {
        double centerx = xmin + width / 2;
        double centery = ymin + height / 2;
        if (width > height) {
            width += height * 2.0;
            height *= 3.0;
        } else {
            height += width * 2.0;
            width *= 3.0;
        }
        double newX = centerx - width / 2 < 0 ? 0 : centerx - width / 2;
        double newY = centery - height / 2 < 0 ? 0 : centery - height / 2;
        double newWidth = newX + width > 1 ? 1 - newX : width;
        double newHeight = newY + height > 1 ? 1 - newY : height;
        return new double[] {newX, newY, newWidth, newHeight};
    }

    public static void main(String[] args) throws Exception {
        String url = "https://resources.djl.ai/images/flight_ticket.jpg";
//        Image img = ImageFactory.getInstance().fromFile(Paths.get("build/files/2233.jpg"));
        Image img = ImageFactory.getInstance().fromUrl(url);
        img.getWrappedImage();

        var criteria1 = Criteria.builder()
                .optEngine("PaddlePaddle")
                .setTypes(Image.class, DetectedObjects.class)
                .optModelUrls("https://resources.djl.ai/test-models/paddleOCR/mobile/det_db.zip")
                .optTranslator(new PpWordDetectionTranslator(new ConcurrentHashMap<String, String>()))
                .build();
        var detectionModel = ModelZoo.loadModel(criteria1);
        var detector = detectionModel.newPredictor();

        var detectedObj = detector.predict(img);
        System.out.println(detectedObj);
        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
        newImage.drawBoundingBoxes(detectedObj);
        newImage.getWrappedImage();
        newImage.save(Files.newOutputStream(Paths.get("build/output").resolve("detected-1.png")), "png");

        List<DetectedObjects.DetectedObject> boxes = detectedObj.items();
        for(int i=0;i<boxes.size();i++) {
            var sample = getSubImage(img, boxes.get(i).getBoundingBox());
            sample.getWrappedImage();
            sample.save(Files.newOutputStream(Paths.get("build/output").resolve("detected-orc"+i+".png")), "png");

            var criteria2 = Criteria.builder()
                    .optEngine("PaddlePaddle")
                    .setTypes(Image.class, Classifications.class)
                    .optModelUrls("https://resources.djl.ai/test-models/paddleOCR/mobile/cls.zip")
                    .optTranslator(new PpWordRotateTranslator())
                    .build();
            var rotateModel = ModelZoo.loadModel(criteria2);
            var rotateClassifier = rotateModel.newPredictor();

            var criteria3 = Criteria.builder()
                    .optEngine("PaddlePaddle")
                    .setTypes(Image.class, String.class)
                    .optModelUrls("https://resources.djl.ai/test-models/paddleOCR/mobile/rec_crnn.zip")
                    .optTranslator(new PpWordRecognitionTranslator())
                    .build();
            var recognitionModel = ModelZoo.loadModel(criteria3);
            var recognizer = recognitionModel.newPredictor();

            System.out.println(rotateClassifier.predict(sample));
            sample.save(Files.newOutputStream(Paths.get("build/output").resolve("detected-orc-result"+i+".png")), "png");
            String res = recognizer.predict(sample);
            System.out.println(res);
        }

    }

    //todo 没有纠偏
    static Image rotateImg(Image image) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray rotated = NDImageUtils.rotate90(image.toNDArray(manager), 1);
            return ImageFactory.getInstance().fromNDArray(rotated);
        }
    }

}
