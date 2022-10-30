use clap::Parser;
use cli::SfmpArgs;
use sfmp::Sfmp;

mod cli;
mod feature_extractor;
mod sfmp;

type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Box<Error>> {
    let args = SfmpArgs::parse();

    let sfmp = Sfmp::new(
        args.path,
        args.num_clusters,
        args.max_n_iterations,
        args.tolerance,
        args.features,
    );

    sfmp.run(args.move_images).unwrap();

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sample() {
        let sfmp = Sfmp::new("samples".to_string(), 2, 300, 1e-4, 512);
        let result = sfmp.run(false).unwrap();

        assert_eq!(result.sum(), 1);
    }
}
