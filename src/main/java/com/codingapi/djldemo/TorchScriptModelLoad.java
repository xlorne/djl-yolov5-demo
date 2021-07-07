package com.codingapi.djldemo;


import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import lombok.extern.slf4j.Slf4j;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * @author lorne
 * @since 1.0.0
 *
 * https://github.com/deepjavalibrary/djl/issues/901
 * https://www.programmersought.com/article/65958865963/
 * https://pytorch.org/docs/1.8.1/jit.html
 */
@Slf4j
public class TorchScriptModelLoad {

    public static void main(String[] args) throws Exception{
        //输入图片的大小 对应导出torchscript的大小
        int imageSize = 640;
        //图片处理步骤
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(imageSize)); //调整尺寸
        pipeline.add(new ToTensor()); //处理为tensor类型
        //定义YoLov5的Translator
        Translator<Image, DetectedObjects> translator =  YoloV5Translator
                .builder()
                .setPipeline(pipeline)
                //labels信息定义
//                .optSynsetArtifactName("coco.names") //数据的labels文件名称
                .optSynset(Arrays.asList("qinggangwa","dapeng","dapengs")) //数据的labels数据
                //预测的最小下限
                .optThreshold(0.5f)
                .build();

        //构建Model Criteria
        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)//图片目标检测类型
                .optModelUrls(Paths.get("build/yolov5").toFile().getPath())//模型的路径
                .optModelName("best.torchscript.pt")//模型的文件名称
                .optTranslator(translator)//设置Translator
                .optProgress(new ProgressBar())//展示加载进度
                .build();
        //加载Model
        ZooModel<Image,DetectedObjects> model = ModelZoo.loadModel(criteria);
        //加载图片
        Image img = ImageFactory.getInstance().fromFile(Paths.get("build/files/DSC00020_0.jpg"));
        //创建预测对象
        Predictor<Image, DetectedObjects> predictor = model.newPredictor();
        //预测图片
        DetectedObjects results = predictor.predict(img);
        System.out.println(results);
        //创建用于保存的预测结果
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);
        //在预测结果上画出标记框
        YoLoImageUtils.drawBoundingBoxes((BufferedImage) img.getWrappedImage(),results);
        //保存文件名称
        Path imagePath = outputDir.resolve("detected-DSC00020_0.png");
        // OpenJDK can't save jpg with alpha channel
        img.save(Files.newOutputStream(imagePath), "png");
        log.info("Detected objects image has been saved in: {}", imagePath);
    }

}
