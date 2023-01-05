from utils.dataloader import dataloader

def main():

    train_data_loader = dataloader()
    train_data = train_data_loader.get_batched_dataset()
    train_data_loader.length()




if __name__ == "__main__":

    main()