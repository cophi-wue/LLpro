import hydra

from event_classify.config import Config
from event_classify.datasets import SimpleJSONEventDataset
from main import get_datasets


@hydra.main(config_name="conf/config")
def main(config: Config):
    new_ds = SimpleJSONEventDataset(hydra.utils.get_original_cwd() + "/data/forTEXT-EvENT_Dataset-e6bc150")
    old_ds = get_datasets(config.dataset)
    breakpoint()
    print(old_ds[0])

main()
# print(dataset[0])
# breakpoint()
# print(dataset[1])

