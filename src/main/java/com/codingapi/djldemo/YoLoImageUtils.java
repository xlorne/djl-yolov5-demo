package com.codingapi.djldemo;

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Landmark;
import ai.djl.modality.cv.output.Mask;
import ai.djl.util.RandomUtils;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;

/**
 * @author lorne
 * @since 1.0.0
 * @see ai.djl.modality.cv.BufferedImageFactory.BufferedImageWrapper
 */
public class YoLoImageUtils {

    private static Color randomColor() {
        return new Color(RandomUtils.nextInt(255));
    }

    public static void drawBoundingBoxes(BufferedImage image, DetectedObjects detections) {
        Graphics2D g = (Graphics2D) image.getGraphics();
        int stroke = 2;
        g.setStroke(new BasicStroke(stroke));
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();

        List<DetectedObjects.DetectedObject> list = detections.items();
        for (DetectedObjects.DetectedObject result : list) {
            String className = result.getClassName();
            BoundingBox box = result.getBoundingBox();
            g.setPaint(randomColor().darker());

            ai.djl.modality.cv.output.Rectangle rectangle = box.getBounds();
            int x = (int) (rectangle.getX() );
            int y = (int) (rectangle.getY() );
            g.drawRect(
                    x,
                    y,
                    (int) (rectangle.getWidth() ),
                    (int) (rectangle.getHeight()));
            drawText(g, className, x, y, stroke, 4);
            // If we have a mask instead of a plain rectangle, draw tha mask
            if (box instanceof Mask) {
                Mask mask = (Mask) box;
                drawMask(image, mask);
            } else if (box instanceof Landmark) {
                drawLandmarks(image, box);
            }
        }
        g.dispose();
    }

    private static void drawLandmarks(BufferedImage image, BoundingBox box) {
        Graphics2D g = (Graphics2D) image.getGraphics();
        g.setColor(new Color(246, 96, 0));
        BasicStroke bStroke = new BasicStroke(4, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER);
        g.setStroke(bStroke);
        for (ai.djl.modality.cv.output.Point point : box.getPath()) {
            g.drawRect((int) point.getX(), (int) point.getY(), 2, 2);
        }
        g.dispose();
    }

    private static void drawText(Graphics2D g, String text, int x, int y, int stroke, int padding) {
        FontMetrics metrics = g.getFontMetrics();
        x += stroke / 2;
        y += stroke / 2;
        int width = metrics.stringWidth(text) + padding * 2 - stroke / 2;
        int height = metrics.getHeight() + metrics.getDescent();
        int ascent = metrics.getAscent();
        java.awt.Rectangle background = new java.awt.Rectangle(x, y, width, height);
        g.fill(background);
        g.setPaint(Color.WHITE);
        g.drawString(text, x + padding, y + ascent);
    }

    private static void drawMask(BufferedImage image, Mask mask) {
        float r = RandomUtils.nextFloat();
        float g = RandomUtils.nextFloat();
        float b = RandomUtils.nextFloat();
        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();
        int x = (int) (mask.getX() * imageWidth);
        int y = (int) (mask.getY() * imageHeight);
        float[][] probDist = mask.getProbDist();
        // Correct some coordinates of box when going out of image
        if (x < 0) {
            x = 0;
        }
        if (y < 0) {
            y = 0;
        }

        BufferedImage maskImage =
                new BufferedImage(
                        probDist.length, probDist[0].length, BufferedImage.TYPE_INT_ARGB);
        for (int xCor = 0; xCor < probDist.length; xCor++) {
            for (int yCor = 0; yCor < probDist[xCor].length; yCor++) {
                float opacity = probDist[xCor][yCor];
                if (opacity < 0.1) {
                    opacity = 0f;
                }
                if (opacity > 0.8) {
                    opacity = 0.8f;
                }
                maskImage.setRGB(xCor, yCor, new Color(r, g, b, opacity).darker().getRGB());
            }
        }
        Graphics2D gR = (Graphics2D) image.getGraphics();
        gR.drawImage(maskImage, x, y, null);
        gR.dispose();
    }
}
