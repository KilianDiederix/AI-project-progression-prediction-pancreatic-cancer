# Paste in slicer, will rename all segmentations (according to Bereska et al. structure)

label_map = {
    1: "Kidney Right",
    2: "Kidney Left",
    3: "Adrenal Gland Right",
    4: "Adrenal Gland Left",
    5: "Spleen",
    6: "Liver",
    7: "Gallbladder",
    8: "Pancreas",
    9: "Duodenum",
    10: "Tumor",
    11: "Aorta",
    12: "Celiac Trunc",
    13: "Hepatic Artery",
    14: "Splenic Artery",
    15: "Superior Mesenteric Artery",
    16: "Inferior Vena Cava",
    17: "Portal Vein",
    18: "Splenic Vein",
    19: "Superior Mesenteric Vein"
}

# Loop through all segmentation nodes and rename segments
for seg_node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
    print(f"Processing: {seg_node.GetName()}")
    segmentation = seg_node.GetSegmentation()
    num_segments = segmentation.GetNumberOfSegments()
    
    for i in range(num_segments):
        segment_id = segmentation.GetNthSegmentID(i)
        label_index = i + 1  # Assumes segment order matches label values
        new_name = label_map.get(label_index, f"Segment {label_index}")
        segmentation.GetSegment(segment_id).SetName(new_name)
        print(f"  Renamed Segment {i+1} → {new_name}")
