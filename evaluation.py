import matplotlib.pyplot as plt
from transformers import TrainerCallback
import datetime
import torch
import os
import json

class EvaluationCallback(TrainerCallback):
    def __init__(self, test_dataset, tokenizer):
        # ... existing code ...
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.epoch = 0
        # Add storage for metrics
        self.eval_losses = []
        self.train_losses = []
        self.epochs = []

    def on_epoch_end(self, args, state, control, model, **kwargs):
        print(f"\nEvaluating model after epoch... {self.epoch}")
    
        # Store the current training loss
        if state.log_history:
            latest_loss = state.log_history[-1].get('loss')
            if latest_loss is not None:
                self.train_losses.append(latest_loss)
                self.epochs.append(self.epoch)

        model.eval()
        total_eval_loss = 0
        num_eval_samples = 0
        
        with torch.no_grad():
            for i in range(min(1, len(self.test_dataset))):
                user_input = self.test_dataset[i]['messages'][1]['content']
                system_prompt = self.test_dataset[i]['messages'][0]['content']
                
                # Prepare the input for the model
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
           
                chat_template_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                model_inputs = self.tokenizer([chat_template_text], 
                                            return_tensors="pt",
                                            ).to(model.device)
                
                # Generate response
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=False      
                )
                
                # Calculate loss
                outputs = model(**model_inputs, labels=model_inputs.input_ids)
                loss = outputs.loss.item()
                total_eval_loss += loss
                num_eval_samples += 1
                
                # Decode and print response
                generated_text = self.tokenizer.batch_decode(
                    [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)],
                    skip_special_tokens=True
                )[0]
                
                print(f"\nTest sample {i+1}:")
                print(f"Input: {user_input}")
                print(f"Output: {generated_text}")
                print(f"Loss: {loss}")
                print("-" * 50)

        # Calculate average evaluation loss
        avg_eval_loss = total_eval_loss / num_eval_samples if num_eval_samples > 0 else 0
        self.eval_losses.append(avg_eval_loss)
        
        # Save metrics
        metrics = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'current_epoch': self.epoch,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create directory if it doesn't exist
        os.makedirs('losses', exist_ok=True)
        
        # Save metrics to JSON
        with open('losses/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create and save plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(range(len(self.eval_losses)), self.eval_losses, 'r-', label='Evaluation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss Over Time')
        plt.grid(True)
        plt.legend()
        plt.savefig('losses/training_progress.png')
        plt.close()
        
        self.epoch += 1
        model.train()