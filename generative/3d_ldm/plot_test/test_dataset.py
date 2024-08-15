import unittest
import torch
from torch.utils.data import DataLoader
from create_dataset import HCPT1wDataset, create_train_val_datasets  # Import your dataset class and function

class TestHCPT1wDataset(unittest.TestCase):
    def setUp(self):
        self.base_dir = '/home/sijun/meow/data/hcp_new/hcp/registered'
        self.train_dataset, self.val_dataset = create_train_val_datasets(self.base_dir)
        self.train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=4, shuffle=False, num_workers=0)

    def test_dataset_sizes(self):
        total_samples = len(self.train_dataset) + len(self.val_dataset)
        self.assertGreater(total_samples, 0, "Dataset is empty")
        self.assertGreater(len(self.train_dataset), 0, "Training dataset is empty")
        self.assertGreater(len(self.val_dataset), 0, "Validation dataset is empty")

    def test_image_shape(self):
        train_sample = next(iter(self.train_loader))
        self.assertEqual(len(train_sample.shape), 5, "Sample should be 5D: (batch, channel, depth, height, width)")
        self.assertEqual(train_sample.shape[0], 4, "Batch size should be 4")
        self.assertEqual(train_sample.shape[1], 1, "Should have 1 channel")

    # def test_image_values(self):
    #     train_sample = next(iter(self.train_loader))
    #     self.assertTrue(torch.is_floating_point(train_sample), "Image should be float tensor")
    #     # self.assertTrue((train_sample >= 0).all() and (train_sample <= 1).all(), "Image values should be normalized between 0 and 1")

    def test_training_loop(self):
        num_epochs = 1
        for epoch in range(num_epochs):
            for batch in self.train_loader:
                print(batch.size())
                self.assertEqual(len(batch.shape), 5, "Batch should be 5D")
                self.assertEqual(batch.shape[0], 4, "Batch size should be 4")
                # Here you could add more assertions or simulate a training step

if __name__ == '__main__':
    unittest.main()
