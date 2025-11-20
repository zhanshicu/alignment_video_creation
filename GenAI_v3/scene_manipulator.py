"""
Scene Manipulator - Main Interface

Manipulate specific scenes in videos to increase/decrease attention-keyword alignment.
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Literal, Optional

from .zero_shot_manipulator import ZeroShotAlignmentManipulator
from .video_processor import VideoProcessor


class SceneManipulator:
    """
    High-level interface for manipulating specific scenes in videos.

    Usage:
        manipulator = SceneManipulator()

        # Increase alignment for scene 6
        manipulator.manipulate_and_replace(
            video_id="123456",
            scene_index=6,
            action="increase",
            keyword="sneakers",
        )
    """

    def __init__(
        self,
        valid_scenes_file: str = "../data/valid_scenes.csv",
        video_dir: str = "../data/data_tiktok",
        output_dir: str = "../outputs/genai_v3/manipulated_videos",
        method: Literal["instruct_pix2pix", "inpainting"] = "instruct_pix2pix",
        device: str = "cuda",
        use_scene_detection: bool = False,
    ):
        """
        Initialize scene manipulator.

        Args:
            valid_scenes_file: Path to valid_scenes.csv
            video_dir: Directory containing videos ({video_id}.mp4)
            output_dir: Directory for output videos
            method: Manipulation method
            device: "cuda" or "cpu"
            use_scene_detection: If True, use automatic scene detection.
                                 If False, estimate from scene count (faster)
        """
        self.valid_scenes_file = valid_scenes_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_scene_detection = use_scene_detection

        # Load valid scenes
        self.scenes_df = pd.read_csv(valid_scenes_file)
        print(f"✓ Loaded {len(self.scenes_df)} valid scenes")

        # Initialize manipulator
        print(f"Initializing {method} model...")
        self.frame_manipulator = ZeroShotAlignmentManipulator(
            method=method,
            device=device,
            torch_dtype=torch.float16,
        )

        # Initialize video processor
        self.video_processor = VideoProcessor(video_dir=video_dir)

    def manipulate_and_replace(
        self,
        video_id: str,
        scene_index: int,
        action: Literal["increase", "decrease"],
        keyword: Optional[str] = None,
        boost_strength: Optional[float] = None,
        num_inference_steps: int = 20,
        blend_frames: int = 5,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Manipulate a specific scene and replace it in the video.

        Args:
            video_id: Video identifier
            scene_index: Scene number (1-indexed, e.g., 6 for scene 6)
            action: "increase" or "decrease" alignment
            keyword: Product keyword (auto-detected if None)
            boost_strength: Manipulation strength (auto-set if None)
                - For "increase": 1.5 (default) means 1.5× more prominent
                - For "decrease": 0.5 (default) means 0.5× less prominent
            num_inference_steps: Diffusion steps (10-50, higher = better quality)
            blend_frames: Number of frames for smooth blending at boundaries
            output_path: Custom output path (auto-generated if None)

        Returns:
            Path to the output video
        """
        print(f"\n{'='*70}")
        print(f"Manipulating Scene {scene_index} in Video {video_id}")
        print(f"Action: {action.upper()}")
        print(f"{'='*70}\n")

        # Step 1: Load scene information
        scene_info = self._get_scene_info(video_id, scene_index)

        # Auto-detect keyword if not provided
        if keyword is None:
            keyword = scene_info['keyword']
            print(f"Auto-detected keyword: '{keyword}'")

        # Auto-set boost strength if not provided
        if boost_strength is None:
            if action == "increase":
                boost_strength = 1.5
            elif action == "decrease":
                boost_strength = 0.5
            else:
                raise ValueError(f"Unknown action: {action}")

        print(f"Boost strength: {boost_strength}× (1.0 = no change)")

        # Step 2: Load original scene image
        scene_image_path = scene_info['scene_image_path']
        scene_image = Image.open(scene_image_path).convert('RGB')
        print(f"\n✓ Loaded scene image: {scene_image_path}")

        # Load keyword mask (if using inpainting)
        keyword_mask = None
        if self.frame_manipulator.method == "inpainting":
            mask_path = scene_info['keyword_mask_path']
            mask = Image.open(mask_path).convert('L')
            keyword_mask = np.array(mask).astype(np.float32) / 255.0
            keyword_mask = (keyword_mask > 0.5).astype(np.float32)
            print(f"✓ Loaded keyword mask: {mask_path}")

        # Step 3: Manipulate scene
        print(f"\nManipulating scene (this may take 20-30 seconds)...")
        manipulated_scene = self.frame_manipulator.manipulate_frame(
            frame=scene_image,
            keyword=keyword,
            boost_level=boost_strength,
            keyword_mask=keyword_mask,
            num_inference_steps=num_inference_steps,
        )
        print(f"✓ Scene manipulation complete")

        # Step 4: Load original video
        print(f"\nLoading original video...")
        video_frames, fps = self.video_processor.load_video(video_id)

        # Step 5: Find scene location in video
        print(f"\nIdentifying scene location in video...")

        if self.use_scene_detection:
            # Use automatic scene detection (more accurate but slower)
            start_frame, end_frame = self.video_processor.get_precise_scene_range(
                frames=video_frames,
                scene_number=scene_index,
            )
        else:
            # Use estimation based on scene count (faster)
            start_frame, end_frame = self.video_processor.get_scene_frame_range(
                video_id=video_id,
                scene_number=scene_index,
                valid_scenes_df=self.scenes_df,
                fps=fps,
            )

        # Step 6: Replace scene in video
        print(f"\nReplacing scene frames in video...")
        edited_frames = self.video_processor.replace_scene_frames(
            video_frames=video_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            replacement_frame=manipulated_scene,
            blend_frames=blend_frames,
        )

        # Step 7: Export edited video
        if output_path is None:
            output_path = self.output_dir / f"{video_id}_scene{scene_index}_{action}.mp4"
        else:
            output_path = Path(output_path)

        print(f"\nExporting edited video...")
        self.video_processor.export_video(
            frames=edited_frames,
            output_path=str(output_path),
            fps=fps,
        )

        print(f"\n{'='*70}")
        print(f"✓ SUCCESS!")
        print(f"{'='*70}")
        print(f"Original video: {video_id}.mp4")
        print(f"Edited scene: Scene {scene_index}")
        print(f"Action: {action} alignment")
        print(f"Output: {output_path}")
        print(f"{'='*70}\n")

        return str(output_path)

    def manipulate_multiple_scenes(
        self,
        video_id: str,
        scene_actions: dict,
        keyword: Optional[str] = None,
        num_inference_steps: int = 20,
        blend_frames: int = 5,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Manipulate multiple scenes in a single video.

        Args:
            video_id: Video identifier
            scene_actions: Dictionary mapping scene_index -> action
                Example: {3: "increase", 6: "increase", 8: "decrease"}
            keyword: Product keyword (auto-detected if None)
            num_inference_steps: Diffusion steps
            blend_frames: Blending frames
            output_path: Custom output path

        Returns:
            Path to the output video
        """
        print(f"\n{'='*70}")
        print(f"Manipulating Multiple Scenes in Video {video_id}")
        print(f"Scenes to edit: {scene_actions}")
        print(f"{'='*70}\n")

        # Load video once
        video_frames, fps = self.video_processor.load_video(video_id)

        # Process each scene
        for scene_index, action in scene_actions.items():
            print(f"\n--- Processing Scene {scene_index} ({action}) ---")

            # Get scene info
            scene_info = self._get_scene_info(video_id, scene_index)

            # Auto-detect keyword if not provided
            if keyword is None:
                kw = scene_info['keyword']
            else:
                kw = keyword

            # Determine boost strength
            boost_strength = 1.5 if action == "increase" else 0.5

            # Load and manipulate scene
            scene_image = Image.open(scene_info['scene_image_path']).convert('RGB')

            keyword_mask = None
            if self.frame_manipulator.method == "inpainting":
                mask = Image.open(scene_info['keyword_mask_path']).convert('L')
                keyword_mask = np.array(mask).astype(np.float32) / 255.0
                keyword_mask = (keyword_mask > 0.5).astype(np.float32)

            manipulated_scene = self.frame_manipulator.manipulate_frame(
                frame=scene_image,
                keyword=kw,
                boost_level=boost_strength,
                keyword_mask=keyword_mask,
                num_inference_steps=num_inference_steps,
            )

            # Find scene location
            if self.use_scene_detection:
                start_frame, end_frame = self.video_processor.get_precise_scene_range(
                    frames=video_frames,
                    scene_number=scene_index,
                )
            else:
                start_frame, end_frame = self.video_processor.get_scene_frame_range(
                    video_id=video_id,
                    scene_number=scene_index,
                    valid_scenes_df=self.scenes_df,
                    fps=fps,
                )

            # Replace in video
            video_frames = self.video_processor.replace_scene_frames(
                video_frames=video_frames,
                start_frame=start_frame,
                end_frame=end_frame,
                replacement_frame=manipulated_scene,
                blend_frames=blend_frames,
            )

        # Export final video
        if output_path is None:
            scene_str = "_".join([f"s{k}{v[0]}" for k, v in scene_actions.items()])
            output_path = self.output_dir / f"{video_id}_{scene_str}.mp4"
        else:
            output_path = Path(output_path)

        self.video_processor.export_video(
            frames=video_frames,
            output_path=str(output_path),
            fps=fps,
        )

        print(f"\n{'='*70}")
        print(f"✓ SUCCESS! Edited {len(scene_actions)} scenes")
        print(f"Output: {output_path}")
        print(f"{'='*70}\n")

        return str(output_path)

    def _get_scene_info(self, video_id: str, scene_index: int) -> dict:
        """Get scene information from valid_scenes.csv."""
        # Find scene
        scene_row = self.scenes_df[
            (self.scenes_df['video_id'] == video_id) &
            (self.scenes_df['scene_number'] == scene_index)
        ]

        if len(scene_row) == 0:
            available_scenes = self.scenes_df[
                self.scenes_df['video_id'] == video_id
            ]['scene_number'].tolist()

            raise ValueError(
                f"Scene {scene_index} not found for video {video_id}. "
                f"Available scenes: {available_scenes}"
            )

        return scene_row.iloc[0].to_dict()

    def preview_scene_location(
        self,
        video_id: str,
        scene_index: int,
        output_dir: Optional[str] = None,
    ):
        """
        Preview where a scene is located in the video (for debugging).

        Saves a visualization showing the scene boundaries.

        Args:
            video_id: Video identifier
            scene_index: Scene number
            output_dir: Output directory for preview images
        """
        import matplotlib.pyplot as plt

        # Load video
        video_frames, fps = self.video_processor.load_video(video_id)

        # Find scene location
        if self.use_scene_detection:
            start_frame, end_frame = self.video_processor.get_precise_scene_range(
                frames=video_frames,
                scene_number=scene_index,
            )
        else:
            start_frame, end_frame = self.video_processor.get_scene_frame_range(
                video_id=video_id,
                scene_number=scene_index,
                valid_scenes_df=self.scenes_df,
                fps=fps,
            )

        # Get scene info
        scene_info = self._get_scene_info(video_id, scene_index)

        # Load reference scene image
        scene_image = Image.open(scene_info['scene_image_path'])

        # Extract frames from video
        before_frame = video_frames[max(0, start_frame - 1)]
        start_video_frame = video_frames[start_frame]
        middle_video_frame = video_frames[(start_frame + end_frame) // 2]
        end_video_frame = video_frames[min(len(video_frames) - 1, end_frame)]
        after_frame = video_frames[min(len(video_frames) - 1, end_frame + 1)]

        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(scene_image)
        axes[0, 0].set_title("Reference Scene Image", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(before_frame)
        axes[0, 1].set_title(f"Before Scene (frame {start_frame-1})", fontsize=12)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(start_video_frame)
        axes[0, 2].set_title(f"Scene Start (frame {start_frame})", fontsize=12, color='green')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(middle_video_frame)
        axes[1, 0].set_title(f"Scene Middle (frame {(start_frame+end_frame)//2})", fontsize=12)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(end_video_frame)
        axes[1, 1].set_title(f"Scene End (frame {end_frame})", fontsize=12, color='red')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(after_frame)
        axes[1, 2].set_title(f"After Scene (frame {end_frame+1})", fontsize=12)
        axes[1, 2].axis('off')

        plt.suptitle(
            f"Scene {scene_index} Location in Video {video_id}\n"
            f"Frames {start_frame}-{end_frame} ({end_frame-start_frame} frames, "
            f"{(end_frame-start_frame)/fps:.1f}s @ {fps:.1f} fps)",
            fontsize=16,
            fontweight='bold'
        )

        plt.tight_layout()

        if output_dir is None:
            output_dir = self.output_dir / "previews"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        preview_path = output_dir / f"{video_id}_scene{scene_index}_preview.png"
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"✓ Preview saved: {preview_path}")
