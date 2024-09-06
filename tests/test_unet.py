import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models.components.yoda.vector_field_regressor import VectorFieldRegressor

device = "cuda" if torch.cuda.is_available() else "cpu"
bs, t, c, h, w = 2, 10, 4, 32, 32
c_time, c_dim, n_token = 1, 256, 64

# Initialize the model
model = VectorFieldRegressor(
    sample_size=[h, w],
    in_channels=c,
    out_channels=c,
    down_block_types=(
        "CrossAttnDownBlockSpatioTemporal",
        "CrossAttnDownBlockSpatioTemporal",
        "CrossAttnDownBlockSpatioTemporal",
        "DownBlockSpatioTemporal",
    ),
    up_block_types=(
        "UpBlockSpatioTemporal",
        "CrossAttnUpBlockSpatioTemporal",
        "CrossAttnUpBlockSpatioTemporal",
        "CrossAttnUpBlockSpatioTemporal",
    ),
    block_out_channels=(64, 128, 256, 512),
    addition_time_embed_dim=128,
    projection_class_embeddings_input_dim=t * 128,
    layers_per_block=2,
    cross_attention_dim=c_dim,
    transformer_layers_per_block=1,
    num_attention_heads=(2, 4, 4, 8),
    num_frames_in_block=[10, 8, 8, 6, 4, 4, 2, 1],
    skip_action=True,
).to(device)
model.eval()

output_shape = (
    (bs, model.num_frames_in_block[-1], c, h, w)
    if model.num_frames_in_block[-1] > 1
    else (bs, c, h, w)
)


@pytest.fixture
def sample_inputs():
    """Fixture to generate sample inputs."""
    input = torch.randn(bs, t, c, h, w).to(device)
    time_step = torch.randint(0, 100, (bs,)).to(device)
    encoder_hidden_states = torch.randn(bs, c_time, n_token, c_dim).to(device)
    additional_time_ids = torch.randn(bs, t).to(device)
    return input, time_step, encoder_hidden_states, additional_time_ids


def test_forward_pass(sample_inputs):
    """Test a standard forward pass with given input."""
    input, time_step, encoder_hidden_states, additional_time_ids = sample_inputs

    with torch.no_grad():
        output = model(
            input,
            timestep=time_step,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=additional_time_ids,
            skip_action=False,
        )

    assert output is not None, "Output should not be None"
    assert (
        output.shape == output_shape
    ), f"output shape {output.shape} should match expected shape {output_shape}"


#
# @pytest.mark.parametrize("batch_size", [1, 2, 8])
# def test_forward_pass_variable_batch_size(batch_size):
#    """Test forward pass with different batch sizes."""
#    input = torch.randn(batch_size, t, c, h, w).to(device)
#    time_step = torch.randint(0, 100, (batch_size,)).to(device)
#    encoder_hidden_states = torch.randn(batch_size, c_time, n_token, c_dim).to(device)
#    additional_time_ids = torch.randn(batch_size, t).to(device)
#
#    with torch.no_grad():
#        output = model(
#            input,
#            timestep=time_step,
#            encoder_hidden_states=encoder_hidden_states,
#            added_time_ids=additional_time_ids,
#            skip_action=False,
#        )
#
#    assert output.shape == input.shape, (
#        f"Output shape {output.shape} should match input shape {input.shape}"
#    )
#
# @pytest.mark.parametrize("temporal_dim", [5, 15, 20])
# def test_forward_pass_variable_temporal_dim(temporal_dim):
#    """Test forward pass with different temporal dimensions."""
#    input = torch.randn(bs, temporal_dim, c, h, w).to(device)
#    time_step = torch.randint(0, 100, (bs,)).to(device)
#    encoder_hidden_states = torch.randn(bs, c_time, n_token, c_dim).to(device)
#    additional_time_ids = torch.randn(bs, temporal_dim).to(device)
#
#    with torch.no_grad():
#        output = model(
#            input,
#            timestep=time_step,
#            encoder_hidden_states=encoder_hidden_states,
#            added_time_ids=additional_time_ids,
#            skip_action=False,
#        )
#
#    assert output.shape == input.shape, (
#        f"Output shape {output.shape} should match input shape {input.shape}"
#    )
#
# @pytest.mark.parametrize("temporal_dim, height, width", [(1, 64, 64), (10, 128, 128), (10, 32, 32)])
# def test_forward_pass_edge_cases(temporal_dim, height, width):
#    """Test forward pass with edge cases like single frame or large spatial dimensions."""
#    input = torch.randn(bs, temporal_dim, c, height, width).to(device)
#    time_step = torch.randint(0, 100, (bs,)).to(device)
#    encoder_hidden_states = torch.randn(bs, c_time, n_token, c_dim).to(device)
#    additional_time_ids = torch.randn(bs, temporal_dim).to(device)
#
#    with torch.no_grad():
#        output = model(
#            input,
#            timestep=time_step,
#            encoder_hidden_states=encoder_hidden_states,
#            added_time_ids=additional_time_ids,
#            skip_action=False,
#        )
#
#    assert output.shape == input.shape, (
#        f"Output shape {output.shape} should match input shape {input.shape}"
#    )


def test_skip_action_enabled(sample_inputs):
    """Test the forward pass with skip_action enabled."""
    input, time_step, encoder_hidden_states, additional_time_ids = sample_inputs

    with torch.no_grad():
        output = model(
            input,
            timestep=time_step,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=additional_time_ids,
            skip_action=True,
        )

    assert output is not None, "Output should not be None when skip_action is enabled"
    assert (
        output.shape == output_shape
    ), f"Output shape {output.shape} should match expected shape {output_shape}"


def test_model_on_cpu(sample_inputs):
    """Test that the model works on CPU."""
    model_cpu = model.to("cpu")
    input, time_step, encoder_hidden_states, additional_time_ids = sample_inputs

    input, time_step, encoder_hidden_states, additional_time_ids = (
        input.cpu(),
        time_step.cpu(),
        encoder_hidden_states.cpu(),
        additional_time_ids.cpu(),
    )

    with torch.no_grad():
        output = model_cpu(
            input,
            timestep=time_step,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=additional_time_ids,
            skip_action=False,
        )

    assert output is not None, "Output should not be None on CPU"
    assert (
        output.shape == output_shape
    ), f"output shape {output.shape} should match expected shape {output_shape}"


def setup_ddp(rank, world_size, backend="nccl"):
    """Set up the DDP environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_ddp():
    """Cleanup the DDP environment."""
    dist.destroy_process_group()


def run_ddp_test(rank, world_size, model_class, sample_inputs):
    """Test function to be run in each DDP process."""
    setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    model = model_class().to(device)
    ddp_model = DDP(model, device_ids=[rank])

    input, time_step, encoder_hidden_states, additional_time_ids = sample_inputs

    # Move inputs to the correct device
    input = input.to(device)
    time_step = time_step.to(device)
    encoder_hidden_states = encoder_hidden_states.to(device)
    additional_time_ids = additional_time_ids.to(device)

    with torch.no_grad():
        output = ddp_model(
            input,
            timestep=time_step,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=additional_time_ids,
            skip_action=False,
        )

    assert output is not None, "Output should not be None in DDP mode"
    assert (
        output.shape == input.shape
    ), f"Output shape {output.shape} should match input shape {input.shape}"

    cleanup_ddp()


def test_ddp_model():
    """Test the model under Distributed Data Parallel (DDP) conditions."""
    world_size = torch.cuda.device_count()
    if world_size < 2:
        pytest.skip("DDP test requires at least 2 GPUs")

    model_class = partial(
        VectorFieldRegressor,
        sample_size=[32, 32],
        in_channels=4,
        out_channels=4,
        down_block_types=(
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types=(
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels=(64, 128, 256, 512),
        addition_time_embed_dim=128,
        projection_class_embeddings_input_dim=10 * 128,
        layers_per_block=2,
        cross_attention_dim=256,
        transformer_layers_per_block=1,
        num_attention_heads=(2, 4, 4, 8),
        num_frames_in_block=[10, 8, 8, 6, 4, 4, 2, 1],
        skip_action=True,
    )

    bs, t, c, h, w = 3, 10, 4, 32, 32
    c_time, c_dim, n_token = 1, 256, 64
    input = torch.randn(bs, t, c, h, w)
    time_step = torch.randint(0, 100, (bs,))
    encoder_hidden_states = torch.randn(bs, c_time, n_token, c_dim)
    additional_time_ids = torch.randn(bs, t)

    sample_inputs = (input, time_step, encoder_hidden_states, additional_time_ids)

    mp.spawn(
        run_ddp_test,
        args=(world_size, model_class, sample_inputs),
        nprocs=world_size,
        join=True,
    )


test_ddp_model()
