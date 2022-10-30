use onnxruntime::{
    environment::Environment,
    ndarray::{Array4, Dim, IxDynImpl},
    session::Session,
    tensor::OrtOwnedTensor,
};
use opencv::{
    core::{Size, Vec3b},
    imgproc,
    prelude::{Mat, MatTraitConstManual},
};

use std::ops::Deref;

type Error = Box<dyn std::error::Error>;

pub struct FeatureExtractor {
    session: Session,
}

impl FeatureExtractor {
    pub fn new() -> Result<FeatureExtractor, Error> {
        let environment = Environment::builder().build()?;
        let model_bytes = include_bytes!("../models/feature_extractor.onnx");
        let session: Session = environment
            .new_session_builder()?
            .with_model_from_memory(model_bytes)?;

        Ok(FeatureExtractor { session })
    }

    pub fn run(
        &mut self,
        filepath: &str,
    ) -> Result<Vec<OrtOwnedTensor<'_, '_, f32, Dim<IxDynImpl>>>, Error> {
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        let bgr = opencv::imgcodecs::imread(filepath, 1)?;
        let mut rgb: Mat = Mat::default();
        imgproc::cvt_color(&bgr, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;
        let mut resized = Mat::default();
        imgproc::resize(
            &rgb,
            &mut resized,
            Size {
                width: 224,
                height: 224,
            },
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let vec = Mat::data_typed::<Vec3b>(&resized).unwrap();

        let array = Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
            (Vec3b::deref(&vec[x + y * 224 as usize])[c] as f32 / 255.0 - mean[c]) / std[c]
        })
        .into();

        let input = vec![array];
        let outputs: Vec<OrtOwnedTensor<f32, _>> = self.session.run(input)?;

        Ok(outputs)
    }
}
