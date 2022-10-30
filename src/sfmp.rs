use std::{
    fs::{self, DirEntry},
    iter::zip,
    path::Path,
    result::Result,
};

use linfa::{
    traits::{Fit, Predict},
    DatasetBase,
};
use linfa_clustering::KMeans;
use onnxruntime::ndarray::{Array2, ArrayBase, Dim, OwnedRepr};
use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256Plus};

use indicatif::ProgressBar;

use crate::feature_extractor::FeatureExtractor;

type Error = Box<dyn std::error::Error>;

pub struct Sfmp {
    path: String,
    num_clusters: usize,
    max_n_iterations: u64,
    tolerance: f32,
    features: usize,
}

impl Sfmp {
    pub fn new(
        path: String,
        num_clusters: usize,
        max_n_iterations: u64,
        tolerance: f32,
        features: usize,
    ) -> Sfmp {
        Sfmp {
            path,
            num_clusters,
            max_n_iterations,
            tolerance,
            features,
        }
    }

    fn get_paths(&self, path: &Path) -> Vec<Result<DirEntry, std::io::Error>> {
        let paths: Vec<Result<DirEntry, std::io::Error>> =
            fs::read_dir(path.as_os_str().to_str().unwrap())
                .unwrap()
                .filter(|x| {
                    [".png", ".jpg", ".jpeg"].iter().any(|e| {
                        x.as_ref()
                            .unwrap()
                            .file_name()
                            .as_os_str()
                            .to_str()
                            .unwrap()
                            .ends_with(e)
                    })
                })
                .into_iter()
                .collect();

        if paths.is_empty() {
            panic!("No images, check path!")
        }

        return paths;
    }

    fn make_dirs(&self, path: &Path) -> Result<(), std::io::Error> {
        for n in 0..self.num_clusters {
            let output_path = path.join(format!("cluster_{}", n));
            fs::create_dir_all(output_path)?;
        }
        Ok(())
    }

    fn make_dataset(
        &self,
        mut feature_extractor: FeatureExtractor,
        paths: &Vec<Result<DirEntry, std::io::Error>>,
    ) -> DatasetBase<
        ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>,
    > {
        let mut dataset_array = Array2::<f32>::default((paths.len(), self.features));
        println!("Extracting features...");
        let pb = ProgressBar::new(paths.len() as u64);
        for (i, path) in paths.iter().enumerate() {
            let result = feature_extractor
                .run(
                    path.as_ref()
                        .unwrap()
                        .path()
                        .into_os_string()
                        .into_string()
                        .unwrap()
                        .as_ref(),
                )
                .unwrap();

            let value = result
                .first()
                .unwrap()
                .to_owned()
                .as_standard_layout()
                .into_shape(self.features)
                .unwrap();

            for (j, v) in value.iter().enumerate() {
                dataset_array[[i, j]] = *v;
            }
            pb.inc(1);
        }

        DatasetBase::from(dataset_array)
    }

    fn make_kmeans(
        &self,
        dataset: &DatasetBase<
            ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
            ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>,
        >,
    ) -> KMeans<f32, linfa_nn::distance::L2Dist> {
        let seed = Xoshiro256Plus::seed_from_u64(0);
        println!("Fitting kmeans...");
        KMeans::params_with_rng(2, seed)
            .max_n_iterations(self.max_n_iterations)
            .tolerance(self.tolerance)
            .fit(&dataset)
            .unwrap()
    }

    fn move_images(
        &self,
        targets: &ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>,
        paths: &Vec<Result<DirEntry, std::io::Error>>,
    ) {
        let res = zip(targets, paths);
        println!("Moving images...");
        let pb = ProgressBar::new(res.len() as u64);
        for (t, p) in res.into_iter() {
            let pp = p.as_ref().unwrap();

            let from = pp.path().into_os_string().into_string().unwrap();

            let filename = pp.file_name().into_string().unwrap().clone();
            let to = pp
                .path()
                .parent()
                .unwrap()
                .join(format!("cluster_{}", t))
                .join(filename)
                .into_os_string()
                .into_string()
                .unwrap();

            // println!("{} => {}", from, to);

            fs::rename(from, to).unwrap();
            pb.inc(1);
        }
    }

    pub fn run(
        &self,
        move_images: bool,
    ) -> Result<ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>, Box<Error>> {
        // println!("{:?}", self.path);
        let feature_extractor = FeatureExtractor::new()?;

        let path = Path::new(&self.path);
        let paths = self.get_paths(path);

        let dataset = self.make_dataset(feature_extractor, &paths);

        let model = self.make_kmeans(&dataset);

        let predicted_dataset = model.predict(dataset);
        let DatasetBase {
            records: _,
            targets,
            ..
        } = predicted_dataset;

        if move_images {
            self.make_dirs(path).unwrap();
            self.move_images(&targets, &paths);
        }

        Ok(targets)
    }
}
