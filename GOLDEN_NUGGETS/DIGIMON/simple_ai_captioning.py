#!/usr/bin/env python3
"""
Simple but effective AI captioning for Digimon images
Uses smaller, more reliable models with better error handling
"""

import os
import time
import random
import requests
from pathlib import Path
from PIL import Image
import re

# Configuration
INPUT_DIR = "digimon_images"
OUTPUT_DIR = "lora_dataset_final"
TARGET_SIZE = (1024, 1024)

def test_internet():
    """Test internet connection."""
    try:
        response = requests.get('https://huggingface.co', timeout=10)
        return response.status_code == 200
    except:
        return False

def load_simple_ai_model():
    """Load a smaller, more reliable AI model."""
    try:
        from transformers import pipeline
        
        print("🔄 Loading smaller AI model for captioning...")
        
        # Try multiple smaller models that are more reliable
        model_options = [
            "microsoft/git-base",
            "nlpconnect/vit-gpt2-image-captioning",
            "Salesforce/blip-image-captioning-base"
        ]
        
        for model_name in model_options:
            try:
                print(f"Trying: {model_name}")
                captioner = pipeline("image-to-text", model=model_name)
                print(f"✅ Successfully loaded: {model_name}")
                return captioner
            except Exception as e:
                print(f"❌ Failed with {model_name}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ AI model loading failed: {e}")
        return None

def generate_ai_caption_simple(image_path, captioner):
    """Generate caption using simple AI model."""
    try:
        result = captioner(image_path)
        caption = result[0]['generated_text']
        
        # Clean up the caption
        caption = caption.lower()
        
        # Make it Digimon-specific
        if 'digimon' not in caption:
            caption = f"a {caption}"
            
        # Remove style references
        style_terms = ['in the style of', 'style', 'art', 'drawing', 'illustration', 'painting']
        for term in style_terms:
            caption = caption.replace(term, '')
            
        # Clean up extra spaces
        caption = ' '.join(caption.split())
        
        return caption.capitalize()
        
    except Exception as e:
        print(f"AI captioning failed: {e}")
        return generate_enhanced_caption(Path(image_path).name)

def generate_enhanced_caption(filename):
    """Enhanced rule-based captioning with 30+ variations."""
    name = Path(filename).stem
    
    # Clean the name
    name = re.sub(r'_(watanabe|modern|anime|other)', '', name, flags=re.IGNORECASE)
    clean_name = re.sub(r'\([^)]*\)', '', name)
    clean_name = re.sub(r'[_-]', ' ', clean_name)
    clean_name = ' '.join(clean_name.split())
    clean_name = clean_name.title()
    
    # Detect characteristics
    stage = detect_stage(name)
    types = detect_types(name)
    
    # Generate diverse captions
    templates = [
        # Basic descriptive
        f"a {stage} {clean_name} digimon {get_type_desc(types)}",
        f"{clean_name}, {get_article(stage)} {stage} digimon {get_detailed_desc(types)}",
        f"a digimon character named {clean_name} in {stage} form {get_visual_features()}",
        
        # Action-oriented
        f"{clean_name} digimon {get_action_phrase(stage)} with {get_physical_features(types)}",
        f"{get_mood()} {clean_name} digimon {get_pose_description()} {get_environment()}",
        
        # Technical
        f"digital monster {clean_name} at {stage} evolutionary level {get_tech_features()}",
        f"{clean_name} species digimon displaying {stage} characteristics {get_unique_traits()}",
        
        # Descriptive variations
        f"{get_adjective(stage)} {clean_name} digimon featuring {get_feature_list(types)}",
        f"{clean_name} in {stage} stage exhibiting {get_behavior(stage)} and {get_appearance(types)}",
        
        # Natural language
        f"this shows {get_article(stage)} {clean_name} digimon with {get_natural_description(types)}",
        f"image of {clean_name} digimon {get_contextual_description(stage, types)}",
        
        # More creative variations
        f"{clean_name} digimon captured in {get_setting()} with {get_attributes(types)}",
        f"{get_size(stage)} {clean_name} digimon {get_movement(stage)} {get_expression()}",
        
        # Additional variations for more diversity
        f"{clean_name} digital creature at {stage} development stage {get_characteristics(types)}",
        f"a {stage} form {clean_name} digimon showcasing {get_special_features(types)}",
        f"{clean_name} species with {get_evolution_traits(stage)} and {get_physical_traits(types)}",
        f"digital being {clean_name} exhibiting {stage} level traits {get_visual_details()}",
        f"{clean_name} monster in {stage} phase with {get_distinctive_elements(types)}",
        f"{get_rarity()} {clean_name} digimon {get_power_level(stage)} {get_combat_features()}",
        f"{clean_name} entity with {get_elemental_traits(types)} and {get_biological_features()}",
        f"{stage} stage {clean_name} digimon {get_transformation_status()} {get_energy_signature()}",
        f"{clean_name} character with {get_color_scheme()} {get_texture_description()}",
        f"{get_temperament()} {clean_name} digimon {get_social_behavior()} {get_habitat_preference()}"
    ]
    
    return random.choice(templates)

# Enhanced helper functions for maximum variety
def detect_stage(name):
    name_lower = name.lower()
    stages = {
        'baby': ['baby', 'fresh', 'in-training', 'digitama'],
        'child': ['child', 'rookie', 'training'],
        'adult': ['adult', 'champion', 'armor'], 
        'perfect': ['perfect', 'ultimate'],
        'mega': ['mega', 'super_ultimate', 'ultra', 'burst']
    }
    for stage, keywords in stages.items():
        if any(keyword in name_lower for keyword in keywords):
            return stage
    return 'standard'

def detect_types(name):
    name_lower = name.lower()
    types = []
    type_map = {
        'dragon': ['dragon', 'dramon', 'greymon', 'wingdramon'],
        'reptile': ['reptile', 'lizard', 'snake', 'saurus', 'tyranno'],
        'mammal': ['mon', 'bear', 'wolf', 'tiger', 'lion', 'leomon', 'garurumon'],
        'bird': ['bird', 'birdramon', 'garudamon', 'hawk', 'eagle'],
        'insect': ['insect', 'kabuterimon', 'stingmon', 'flymon', 'beemon'],
        'aquatic': ['fish', 'whamon', 'seadramon', 'octomon', 'marine'],
        'plant': ['plant', 'flower', 'wood', 'forest', 'leaf', 'palmon'],
        'machine': ['machine', 'cyber', 'metal', 'robot', 'andromon', 'guardromon'],
        'holy': ['holy', 'angel', 'divine', 'seraphimon', 'cherubimon', 'angemon'],
        'dark': ['dark', 'evil', 'demon', 'devil', 'neodevimon', 'myotismon']
    }
    for digimon_type, keywords in type_map.items():
        if any(keyword in name_lower for keyword in keywords):
            types.append(digimon_type)
    return types if types else ['creature']

def get_type_desc(types):
    descriptors = {
        'dragon': "with draconic scales and fierce appearance",
        'mammal': "with animalistic features and natural grace", 
        'bird': "displaying avian characteristics and winged form",
        'machine': "constructed with mechanical components",
        'holy': "radiating divine energy and celestial beauty",
        'dark': "emanating dark power and menacing presence",
        'insect': "with insectoid anatomy and chitinous exoskeleton",
        'aquatic': "adapted for aquatic environments",
        'plant': "with botanical features and natural growth",
        'reptile': "featuring reptilian scales and cold-blooded nature"
    }
    for t in types:
        if t in descriptors:
            return descriptors[t]
    return "with unique digital characteristics"

def get_article(stage):
    return "an" if stage in ['adult', 'ultimate'] else "a"

def get_detailed_desc(types):
    descriptions = {
        'dragon': "featuring scaled armor, sharp claws, and powerful wings",
        'mammal': "with fur-covered body, expressive eyes, and animalistic grace",
        'bird': "displaying feathered wings, sharp talons, and avian agility",
        'machine': "constructed from metallic alloys with precision engineering",
        'holy': "surrounded by divine light with angelic wings and pure energy",
        'dark': "wreathed in shadow with demonic features and ominous aura"
    }
    for t in types:
        if t in descriptions:
            return descriptions[t]
    return "with complex digital structure and unique morphology"

def get_visual_features():
    return random.choice([
        "dynamic combat stance and intense expression",
        "detailed surface textures and intricate design patterns",
        "vibrant color palette and striking visual presence",
        "complex anatomical structure with unique morphological features",
        "impressive physical stature and commanding digital presence"
    ])

def get_action_phrase(stage):
    actions = {
        'baby': "playfully exploring",
        'child': "energetically moving", 
        'adult': "powerfully standing",
        'perfect': "confidently posing",
        'mega': "majestically dominating"
    }
    return actions.get(stage, "digitally existing")

def get_physical_features(types):
    features = []
    if 'dragon' in types:
        features.extend(["scaled hide", "draconic wings", "reptilian eyes", "sharp teeth", "powerful tail"])
    if 'mammal' in types:
        features.extend(["thick fur", "animalistic muzzle", "clawed paws", "expressive face", "muscular build"])
    if 'bird' in types:
        features.extend(["feathered wings", "avian beak", "taloned feet", "light frame", "aerodynamic shape"])
    if not features:
        features = ["unique markings", "distinctive coloration", "characteristic posture", "signature features"]
    return ", ".join(random.sample(features, min(3, len(features))))

def get_mood():
    return random.choice(["determined", "focused", "alert", "curious", "confident", "playful", "serious"])

def get_pose_description():
    return random.choice(["in combat stance", "ready for action", "displaying power", "showing agility", "demonstrating strength"])

def get_environment():
    return random.choice(["digital landscape", "virtual environment", "cyberspace backdrop", "digital world setting"])

def get_tech_features():
    return random.choice(["advanced circuitry", "digital energy patterns", "complex data streams", "evolved programming"])

def get_unique_traits():
    return random.choice(["unique genetic code", "special abilities", "distinctive capabilities", "rare attributes"])

def get_adjective(stage):
    adjectives = {
        'baby': ["tiny", "adorable", "developing", "beginning"],
        'child': ["young", "energetic", "growing", "promising"],
        'adult': ["powerful", "mature", "strong", "capable"],
        'perfect': ["evolved", "enhanced", "advanced", "perfected"],
        'mega': ["ultimate", "supreme", "final", "peak"]
    }
    return random.choice(adjectives.get(stage, ["impressive", "notable", "remarkable"]))

def get_feature_list(types):
    features = []
    for t in types:
        if t == 'dragon': features.extend(["scaled armor", "draconic features", "reptilian characteristics"])
        if t == 'mammal': features.extend(["mammalian traits", "animalistic features", "fur covering"])
        if t == 'bird': features.extend(["avian attributes", "winged form", "lightweight build"])
    if not features: features = ["unique characteristics", "distinctive features", "special attributes"]
    return ", ".join(random.sample(features, min(2, len(features))))

def get_behavior(stage):
    behaviors = {
        'baby': ["curious exploration", "playful behavior"],
        'child': ["energetic movement", "rapid learning"],
        'adult': ["confident action", "strategic thinking"],
        'perfect': ["masterful control", "advanced capabilities"],
        'mega': ["ultimate power", "supreme dominance"]
    }
    return random.choice(behaviors.get(stage, ["digital behavior", "characteristic actions"]))

def get_appearance(types):
    appearances = {
        'dragon': "imposing draconic appearance",
        'mammal': "natural animalistic appearance", 
        'bird': "graceful avian appearance",
        'machine': "precise mechanical appearance",
        'holy': "radiant divine appearance",
        'dark': "intimidating dark appearance"
    }
    for t in types:
        if t in appearances:
            return appearances[t]
    return "unique digital appearance"

def get_natural_description(types):
    descriptors = {
        'dragon': "scaled body and fierce expression",
        'mammal': "fur-covered form and animal grace",
        'bird': "feathered wings and sharp features",
        'machine': "metallic construction and robotic precision"
    }
    for t in types:
        if t in descriptors:
            return descriptors[t]
    return "distinctive features and unique design"

def get_contextual_description(stage, types):
    return f"at {stage} evolutionary stage with {get_type_desc(types)}"

def get_setting():
    return random.choice(["digital forest", "virtual arena", "cyber landscape", "data stream environment"])

def get_attributes(types):
    attrs = []
    if 'dragon' in types: attrs.append("draconic power")
    if 'mammal' in types: attrs.append("animal strength")
    if 'bird' in types: attrs.append("avian speed")
    if not attrs: attrs = ["digital energy", "unique capabilities"]
    return " and ".join(attrs)

def get_size(stage):
    sizes = {
        'baby': "small", 'child': "medium-sized", 
        'adult': "large", 'perfect': "impressively large",
        'mega': "massive"
    }
    return sizes.get(stage, "digitally sized")

def get_movement(stage):
    movements = {
        'baby': "playfully moving", 'child': "energetically active",
        'adult': "powerfully positioned", 'perfect': "confidently stationed",
        'mega': "dominantly placed"
    }
    return movements.get(stage, "digitally present")

def get_expression():
    return random.choice(["intense gaze", "focused expression", "alert demeanor", "determined look"])

def get_characteristics(types):
    chars = []
    for t in types:
        if t == 'dragon': chars.append("draconic heritage")
        if t == 'mammal': chars.append("mammalian instincts")
        if t == 'bird': chars.append("avian agility")
    if not chars: chars = ["digital nature", "unique properties"]
    return " and ".join(chars)

def get_special_features(types):
    features = []
    for t in types:
        if t == 'dragon': features.append("fire-breathing capability")
        if t == 'machine': features.append("advanced circuitry")
        if t == 'holy': features.append("divine energy")
    if not features: features = ["special abilities", "unique powers"]
    return ", ".join(features[:2])

def get_evolution_traits(stage):
    traits = {
        'baby': "basic digital form", 'child': "developing capabilities",
        'adult': "mature power", 'perfect': "advanced evolution",
        'mega': "ultimate transformation"
    }
    return traits.get(stage, "digital development")

def get_physical_traits(types):
    traits = []
    for t in types:
        if t == 'dragon': traits.append("scaled physique")
        if t == 'mammal': traits.append("muscular build")
        if t == 'bird': traits.append("lightweight frame")
    if not traits: traits = ["physical form", "bodily structure"]
    return " and ".join(traits)

def get_visual_details():
    return random.choice(["intricate design patterns", "detailed surface textures", "complex color schemes", "unique morphological features"])

def get_distinctive_elements(types):
    elements = []
    for t in types:
        if t == 'dragon': elements.append("draconic elements")
        if t == 'machine': elements.append("mechanical components")
        if t == 'holy': elements.append("divine aspects")
    if not elements: elements = ["distinct features", "unique characteristics"]
    return " and ".join(elements)

def get_rarity():
    return random.choice(["rare", "uncommon", "unique", "special"])

def get_power_level(stage):
    levels = {
        'baby': "basic power", 'child': "growing strength",
        'adult': "mature strength", 'perfect': "advanced power",
        'mega': "ultimate strength"
    }
    return levels.get(stage, "digital power")

def get_combat_features():
    return random.choice(["combat-ready stance", "battle-hardened appearance", "warrior physique", "fighter attributes"])

def get_elemental_traits(types):
    elements = []
    if 'dragon' in types: elements.append("fiery essence")
    if 'aquatic' in types: elements.append("water affinity")
    if 'plant' in types: elements.append("earth connection")
    if 'machine' in types: elements.append("electric energy")
    if not elements: elements = ["elemental properties", "energy signature"]
    return " and ".join(elements)

def get_biological_features():
    return random.choice(["organic composition", "biological structure", "living tissue", "cellular makeup"])

def get_transformation_status():
    return random.choice(["mid-transformation", "fully evolved", "stable form", "evolving state"])

def get_energy_signature():
    return random.choice(["high energy output", "unique energy pattern", "powerful aura", "distinct energy signature"])

def get_color_scheme():
    return random.choice(["vibrant color scheme", "unique coloration", "distinct color pattern", "special color palette"])

def get_texture_description():
    return random.choice(["detailed texture", "complex surface", "intricate patterning", "unique material appearance"])

def get_temperament():
    return random.choice(["calm", "aggressive", "playful", "serious", "curious", "protective"])

def get_social_behavior():
    return random.choice(["solitary nature", "pack behavior", "social interaction", "independent attitude"])

def get_habitat_preference():
    return random.choice(["forest habitat", "urban environment", "digital realm", "natural setting"])

def main():
    """Main function for simple AI captioning."""
    print("=== Simple AI Digimon Captioning ===")
    print("🔄 Setting up AI model...")
    
    # Test internet first
    if not test_internet():
        print("❌ No internet connection - using enhanced rule-based captions")
        use_ai = False
    else:
        captioner = load_simple_ai_model()
        use_ai = captioner is not None
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' not found!")
        return
    
    image_files = [f for f in os.listdir(INPUT_DIR) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ No images found in '{INPUT_DIR}'!")
        return
    
    print(f"📸 Processing {len(image_files)} images...")
    
    successful = 0
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(INPUT_DIR, filename)
        output_filename = f"{Path(filename).stem}.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"\n[{i}/{len(image_files)}] {filename}")
        
        # Generate caption
        if use_ai:
            caption = generate_ai_caption_simple(input_path, captioner)
            print(f"🤖 AI: {caption}")
        else:
            caption = generate_enhanced_caption(filename)
            print(f"📝 Enhanced: {caption}")
        
        # Save caption
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(caption)
        
        successful += 1
    
    print(f"\n✅ Completed: {successful}/{len(image_files)} captions generated")
    print(f"📁 Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"🤖 AI Status: {'Enabled' if use_ai else 'Enhanced rule-based'}")

if __name__ == "__main__":
    main()
