use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about)]
pub struct SfmpArgs {
    pub path: String,

    #[clap(short, long, value_parser, default_value_t = 2)]
    pub num_clusters: usize,

    #[clap(short, long, value_parser, default_value_t = 300)]
    pub max_n_iterations: u64,

    #[clap(short, long, value_parser, default_value_t = 1e-4)]
    pub tolerance: f32,

    #[clap(short, long, value_parser, default_value_t = 224)]
    pub input_size: usize,

    #[clap(short, long, value_parser, default_value_t = 512)]
    pub features: usize,

    #[clap(short, long, value_parser, default_value_t = true)]
    pub move_images: bool,
}
