import torch
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util.train import train_image

class GroundingDINOModel:
    def __init__(self, config_path, weights_path, device="cuda"):
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device
        self.model = None
        
    def load_pretrained(self, freeze_heads=True):
        """Загружает предобученную модель"""
        print(f"📥 Загрузка модели из {self.weights_path}")
        self.model = load_model(self.config_path, self.weights_path)
        self.model = self.model.to(self.device)
        
        if freeze_heads:
            self._freeze_layers()
        
        return self.model
    
    def load_trained(self, trained_weights_path):
        """Загружает дообученные веса"""
        print(f"📥 Загрузка обученных весов из {trained_weights_path}")
        self.model.load_state_dict(
            torch.load(trained_weights_path, map_location=self.device)
        )
        self.model.eval()
        print("✅ Модель загружена и переведена в режим инференса")
        return self.model
    
    def _freeze_layers(self):
        """Заморозка всех слоев кроме голов"""
        for param in self.model.parameters():
            param.requires_grad = False
            
        trainable_params = []
        for name, param in self.model.named_parameters():
            if "class_embed" in name or "bbox_embed" in name:
                param.requires_grad = True
                trainable_params.append(name)
                print(f"  🔓 TRAIN: {name}")
        
        # Статистика
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"\n📊 Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def train(self, ann_file, images_dir, epochs=3, lr=5e-6, 
              save_path="weights/model_", save_epoch=1):
        """Обучение модели"""
        from src.data.grounding_dino_dataset import GroundingDINODataset
        import torch.optim as optim
        import random
        
        # Создаем датасет
        dataset = GroundingDINODataset(ann_file, images_dir, split='train')
        print(f"🚀 Запуск обучения на {len(dataset)} изображениях")
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr
        )
        scaler = torch.amp.GradScaler('cuda')
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            processed = 0
            
            items = dataset.get_all_items()
            random.shuffle(items)
            
            for idx, (image_path, vals) in enumerate(items):
                try:
                    image_source, image = load_image(image_path)
                    
                    loss = self._train_step(
                        image_source, image, 
                        vals['boxes'], vals['captions'],
                        optimizer, scaler
                    )
                    
                    total_loss += loss
                    processed += 1
                    
                    if idx % 20 == 0:
                        print(f"[Epoch {epoch+1}] Step {idx}/{len(items)} Loss: {loss:.4f}")
                        
                except Exception as e:
                    print(f"⚠️ Ошибка: {image_path} - {e}")
                    continue
            
            if processed > 0:
                avg_loss = total_loss / processed
                print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Обработано: {processed}")
            
            if (epoch + 1) % save_epoch == 0:
                save_file = f"{save_path}{epoch+1}.pth"
                torch.save(self.model.state_dict(), save_file)
                print(f"💾 Saved: {save_file}")
    
    def _train_step(self, image_source, image, boxes, captions, optimizer, scaler):
        """Один шаг обучения"""
        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda"):
            loss = train_image(
                model=self.model,
                image_source=image_source,
                image=image,
                caption_objects=captions,
                box_target=boxes,
            )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        return loss.item()
    
    def predict_image(self, image_path, caption, box_threshold=0.3, text_threshold=0.25):
        """Инференс на одном изображении"""
        image_source, image_tensor = load_image(image_path)
        
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        
        return image_source, boxes, logits, phrases