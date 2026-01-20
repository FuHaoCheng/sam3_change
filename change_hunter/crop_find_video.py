import torch

from sam3 import build_sam3_video_model


class SegEarthOV3VideoTracker:
    def __init__(
        self,
        checkpoint_path,
        bpe_path,
        device=None,
        has_presence_token=True,
        geo_encoder_use_img_cross_attn=True,
        strict_state_dict_loading=True,
        apply_temporal_disambiguation=True,
        video_loader_type="cv2",
        async_loading_frames=False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.video_loader_type = video_loader_type
        self.async_loading_frames = async_loading_frames
        self.model = build_sam3_video_model(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            has_presence_token=has_presence_token,
            geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
            strict_state_dict_loading=strict_state_dict_loading,
            apply_temporal_disambiguation=apply_temporal_disambiguation,
            device=self.device,
        ).eval()

    def start_session(self, resource_path):
        return self.model.init_state(
            resource_path=resource_path,
            async_loading_frames=self.async_loading_frames,
            video_loader_type=self.video_loader_type,
        )

    @torch.inference_mode()
    def add_text_prompt(self, inference_state, frame_idx, text_prompt):
        return self.model.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text_prompt,
            boxes_xywh=None,
            box_labels=None,
        )

    @torch.inference_mode()
    def propagate(
        self,
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=None,
        propagation_direction="both",
    ):
        if propagation_direction not in {"both", "forward", "backward"}:
            raise ValueError(
                "propagation_direction must be one of: both, forward, backward"
            )

        if propagation_direction in {"both", "forward"}:
            for frame_idx, outputs in self.model.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=False,
            ):
                yield {"frame_index": frame_idx, "outputs": outputs}

        if propagation_direction in {"both", "backward"}:
            for frame_idx, outputs in self.model.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=True,
            ):
                yield {"frame_index": frame_idx, "outputs": outputs}

    def track_video(
        self,
        resource_path,
        text_prompt,
        start_frame_idx=0,
        max_frame_num_to_track=None,
        propagation_direction="both",
    ):
        inference_state = self.start_session(resource_path)
        _, prompt_outputs = self.add_text_prompt(
            inference_state=inference_state,
            frame_idx=start_frame_idx,
            text_prompt=text_prompt,
        )

        results = [{"frame_index": start_frame_idx, "outputs": prompt_outputs}]
        results.extend(
            list(
                self.propagate(
                    inference_state=inference_state,
                    start_frame_idx=start_frame_idx,
                    max_frame_num_to_track=max_frame_num_to_track,
                    propagation_direction=propagation_direction,
                )
            )
        )
        return results
