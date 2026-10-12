#!/usr/bin/env python3
# Copyright 2024-2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Unit tests for FilesystemClientCharm."""

import json
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
from charm import FilesystemClientCharm
from charms.filesystem_client.v0.filesystem_info import NfsInfo
from charms.filesystem_client.v0.mount_info import MountInfo
from ops import testing
from utils.manager import Error


@pytest.fixture(scope="function")
def ctx() -> testing.Context[FilesystemClientCharm]:
    """Mock `FilesystemClientCharm` context."""
    return testing.Context(FilesystemClientCharm)


@pytest.fixture(scope="function")
def mock_mounts_manager() -> MagicMock:
    """Mock `charm.MountsManager`."""
    with patch("charm.MountsManager") as manager_cls:
        mock_manager = MagicMock()
        manager_cls.return_value = mock_manager
        yield mock_manager


@pytest.mark.parametrize(
    "is_setup", (pytest.param(True, id="is setup"), pytest.param(False, id="is not setup"))
)
def test_manager_mounted_false_if_exc(
    ctx: testing.Context, is_setup: bool, mock_mounts_manager: MagicMock
) -> None:
    """mounted=false is written in the case of a raised exception."""
    mount_rel = testing.SubordinateRelation(endpoint="mount")
    state_in = testing.State(relations={mount_rel})

    mock_mounts_manager.supported.return_value = True
    mock_mounts_manager.is_setup.return_value = is_setup
    mock_mounts_manager.setup.side_effect = Error("apt failed")
    mock_mounts_manager.mounts.side_effect = RuntimeError("unexpected")

    state_out = ctx.run(ctx.on.config_changed(), state_in)

    assert state_out.get_relation(mount_rel.id).local_unit_data["mounted"] == "false"
    assert state_out.unit_status == testing.BlockedStatus(
        "Failed to mount filesystems. See `juju debug-log` for details"
    )


def test_successful_mount_sets_mounted_true(
    ctx: testing.Context, mock_mounts_manager: MagicMock
) -> None:
    """mounted=true is written and ActiveStatus is set after a successful mount."""
    mount_rel = testing.SubordinateRelation(
        endpoint="mount",
        remote_app_data={
            k: json.dumps(v) for k, v in asdict(MountInfo(mountpoint="/mnt/nfs")).items()
        },
    )
    fs_rel = testing.Relation(
        endpoint="filesystem",
        remote_app_data={"endpoint": "nfs://(192.168.1.1)/srv"},
    )
    state_in = testing.State(
        leader=True,
        relations={mount_rel, fs_rel},
    )

    mock_mounts_manager.supported.return_value = True
    mock_mounts_manager.is_setup.return_value = True
    mock_mounts_manager.mounts = MagicMock()

    state_out = ctx.run(ctx.on.config_changed(), state_in)

    assert state_out.get_relation(mount_rel.id).local_unit_data["mounted"] == "true"
    assert state_out.unit_status == testing.ActiveStatus("Mounted filesystem at `/mnt/nfs`")


@pytest.mark.parametrize(
    "enable_lustre",
    (pytest.param(True, id="lustre enabled"), pytest.param(False, id="lustre disabled")),
)
def test_enable_lustre_sets_manager_flag(
    ctx: testing.Context, enable_lustre: bool, mock_mounts_manager: MagicMock
) -> None:
    """The enable-lustre config determines if the mount manager enables lustre support."""
    state_in = testing.State(config={"enable-lustre": enable_lustre})
    mock_mounts_manager.supported.return_value = True
    mock_mounts_manager.is_setup.return_value = True
    mock_mounts_manager.mounts = MagicMock()

    state_out = ctx.run(ctx.on.config_changed(), state_in)
    assert mock_mounts_manager.enable_lustre == enable_lustre
    assert state_out.unit_status == testing.BlockedStatus(
        "Missing `mountpoint` config or `mount` integration"
    )


def test_lnet_networks_config_parsed(ctx: testing.Context, mock_mounts_manager: MagicMock) -> None:
    """The lnet-networks config is parsed before being passed to the manager."""
    state_in = testing.State(
        config={
            "enable-lustre": True,
            "lnet-networks": "tcp=eth0; o2ib=ib0,ib1",
        }
    )
    mock_mounts_manager.supported.return_value = True
    mock_mounts_manager.is_setup.return_value = True
    mock_mounts_manager.mounts = MagicMock()

    ctx.run(ctx.on.config_changed(), state_in)

    assert mock_mounts_manager.lnet_networks == {
        "tcp": ["eth0"],
        "o2ib": ["ib0", "ib1"],
    }


def test_lxd_container_blocked(ctx: testing.Context, mock_mounts_manager: MagicMock) -> None:
    """The charm blocks on LXD containers instead of attempting mounts that cannot work."""
    mock_mounts_manager.supported.return_value = False

    state_out = ctx.run(ctx.on.config_changed(), testing.State())

    assert state_out.unit_status == testing.BlockedStatus(
        "Cannot mount filesystems on LXD containers"
    )
    mock_mounts_manager.setup.assert_not_called()


@pytest.mark.parametrize(
    "leader", (pytest.param(True, id="leader"), pytest.param(False, id="not leader"))
)
def test_no_filesystem_integration_blocked(
    ctx: testing.Context, mock_mounts_manager: MagicMock, leader: bool
) -> None:
    """Without a filesystem integration the unit blocks and the leader reflects it on the app."""
    state_in = testing.State(leader=leader, config={"mountpoint": "/nfs"})
    mock_mounts_manager.supported.return_value = True
    mock_mounts_manager.is_setup.return_value = True

    state_out = ctx.run(ctx.on.config_changed(), state_in)

    expected = testing.BlockedStatus("Waiting for an integration with a filesystem provider")
    assert state_out.unit_status == expected
    assert state_out.app_status == (expected if leader else testing.UnknownStatus())


@pytest.mark.parametrize(
    "leader", (pytest.param(True, id="leader"), pytest.param(False, id="not leader"))
)
def test_integrated_with_provider_app_status(
    ctx: testing.Context, mock_mounts_manager: MagicMock, leader: bool
) -> None:
    """The leader reports the integrated filesystem type on the application status."""
    fs_rel = testing.Relation(
        endpoint="filesystem",
        remote_app_data={"endpoint": "nfs://(192.168.1.1)/srv"},
    )
    state_in = testing.State(
        leader=leader,
        relations={fs_rel},
        config={"mountpoint": "/nfs"},
    )
    mock_mounts_manager.supported.return_value = True
    mock_mounts_manager.is_setup.return_value = True

    state_out = ctx.run(ctx.on.config_changed(), state_in)

    assert state_out.unit_status == testing.ActiveStatus("Mounted filesystem at `/nfs`")
    expected = (
        testing.ActiveStatus("Integrated with `nfs` provider")
        if leader
        else testing.UnknownStatus()
    )
    assert state_out.app_status == expected


def test_multiple_mount_relations_blocked(
    ctx: testing.Context, mock_mounts_manager: MagicMock
) -> None:
    """More than one `mount` relation is rejected: a unit can only mount one filesystem."""
    state_in = testing.State(
        relations={
            testing.SubordinateRelation(endpoint="mount"),
            testing.SubordinateRelation(endpoint="mount"),
        }
    )
    mock_mounts_manager.supported.return_value = True
    mock_mounts_manager.is_setup.return_value = True

    state_out = ctx.run(ctx.on.config_changed(), state_in)

    assert state_out.unit_status == testing.BlockedStatus(
        "Cannot mount using more than one relation at the same time"
    )


def test_mount_relation_without_mountpoint_waiting(
    ctx: testing.Context, mock_mounts_manager: MagicMock
) -> None:
    """The charm waits when the mount requirer has not yet provided a mountpoint."""
    mount_rel = testing.SubordinateRelation(endpoint="mount", remote_app_data={})
    state_in = testing.State(relations={mount_rel})
    mock_mounts_manager.supported.return_value = True
    mock_mounts_manager.is_setup.return_value = True

    state_out = ctx.run(ctx.on.config_changed(), state_in)

    assert state_out.unit_status == testing.WaitingStatus(
        "Waiting for mountpoint from `mount` integration"
    )


def test_mountpoint_config_and_mount_relation_blocked(
    ctx: testing.Context, mock_mounts_manager: MagicMock
) -> None:
    """Setting `mountpoint` config while integrated with a mount requirer is rejected."""
    mount_rel = testing.SubordinateRelation(
        endpoint="mount",
        remote_app_data={
            k: json.dumps(v) for k, v in asdict(MountInfo(mountpoint="/mnt/nfs")).items()
        },
    )
    state_in = testing.State(relations={mount_rel}, config={"mountpoint": "/nfs"})
    mock_mounts_manager.supported.return_value = True
    mock_mounts_manager.is_setup.return_value = True

    state_out = ctx.run(ctx.on.config_changed(), state_in)

    assert state_out.unit_status == testing.BlockedStatus(
        "Cannot mount using both the `mountpoint` config and the `mount` integration"
    )


@pytest.mark.parametrize(
    "config,expected_options",
    (
        pytest.param({}, ["exec", "suid", "dev", "rw"], id="defaults"),
        pytest.param(
            {"noexec": True, "nosuid": True, "nodev": True, "read-only": True},
            ["noexec", "nosuid", "nodev", "ro"],
            id="hardened",
        ),
    ),
)
def test_config_mapped_to_mount_options(
    ctx: testing.Context,
    mock_mounts_manager: MagicMock,
    config: dict,
    expected_options: list[str],
) -> None:
    """The mount security config options are mapped to the options handed to the manager.

    Failure mode: a flipped mapping would silently mount the filesystem with the
    execution, suid, or device access that the operator explicitly disabled.
    """
    fs_rel = testing.Relation(
        endpoint="filesystem",
        remote_app_data={"endpoint": "nfs://(192.168.1.1)/srv"},
    )
    state_in = testing.State(relations={fs_rel}, config={"mountpoint": "/nfs", **config})
    mock_mounts_manager.supported.return_value = True
    mock_mounts_manager.is_setup.return_value = True

    state_out = ctx.run(ctx.on.config_changed(), state_in)

    mounts = mock_mounts_manager.mounts.return_value.__enter__.return_value
    mounts.add.assert_called_once_with(
        info=NfsInfo(hostname="192.168.1.1", port=None, path="/srv", version=None),
        mountpoint="/nfs",
        options=expected_options,
    )
    assert state_out.unit_status == testing.ActiveStatus("Mounted filesystem at `/nfs`")
