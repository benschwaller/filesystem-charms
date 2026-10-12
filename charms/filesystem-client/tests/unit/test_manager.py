# Copyright 2024-2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Unit tests for `utils.manager.MountsManager`."""

import pathlib
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from charmed_hpc_libs.errors import SystemdError, UnknownVirtualizationStateError
from charmlibs import apt
from charms.filesystem_client.v0.filesystem_info import (
    CephfsInfo,
    FilesystemInfo,
    LustreInfo,
    NfsInfo,
)
from lustre_ops.errors import LNetError, RepositoryError
from pytest_mock import MockerFixture
from utils.constants import BASE_PACKAGES, LUSTRE_PACKAGES
from utils.manager import Error, Mounts, MountsManager, lnet, ppa


@pytest.fixture(scope="function")
def manager(mocker: MockerFixture, tmp_path: pathlib.Path) -> MountsManager:
    """Mock MountsManager with package and autofs concerns stubbed out."""
    charm = MagicMock()
    charm.unit.name = "filesystem-client/0"
    mgr = MountsManager(charm)
    # Stub out package installation and autofs file handling so setup() reaches the lnet path.
    mocker.patch.object(mgr, "_packages", return_value=[])
    mocker.patch.object(mgr, "_master_file", tmp_path / "master")
    mocker.patch.object(mgr, "_autofs_file", tmp_path / "autofs")
    return mgr


@pytest.fixture(scope="function")
def raw_manager(mocker: MockerFixture, tmp_path: pathlib.Path) -> MountsManager:
    """MountsManager with only the autofs file paths redirected to a temporary directory."""
    charm = MagicMock()
    charm.unit.name = "filesystem-client/0"
    mgr = MountsManager(charm)
    mocker.patch.object(mgr, "_master_file", tmp_path / "master")
    mocker.patch.object(mgr, "_autofs_file", tmp_path / "autofs")
    return mgr


class TestError:
    """The manager `Error` type."""

    def test_message(self) -> None:
        """`message` carries the original error text.

        `setup()` relies on it when wrapping apt errors, so a regression here
        would surface empty error messages to the operator.
        """
        assert Error("apt failed").message == "apt failed"


class TestSupported:
    """Virtualization gating in MountsManager.supported()."""

    @pytest.mark.parametrize(
        "is_container,expected",
        (
            pytest.param(True, False, id="container"),
            pytest.param(False, True, id="machine"),
        ),
    )
    def test_virtualization_state(
        self, raw_manager: MountsManager, mocker: MockerFixture, is_container: bool, expected: bool
    ) -> None:
        """Containers cannot mount filesystems; machines can."""
        mocker.patch("utils.manager.is_container", return_value=is_container)

        assert raw_manager.supported() is expected

    def test_unknown_virtualization_fails_open(
        self, raw_manager: MountsManager, mocker: MockerFixture
    ) -> None:
        """An undetectable virtualization state is treated as supported.

        Pins the deliberate fail-open: a detection failure must not block every
        unit in the deployment.
        """
        mocker.patch(
            "utils.manager.is_container",
            side_effect=UnknownVirtualizationStateError("unknown state"),
        )

        assert raw_manager.supported() is True


class TestPackages:
    """Package resolution in MountsManager._packages()."""

    def test_base_packages_cached(self, raw_manager: MountsManager, mocker: MockerFixture) -> None:
        """Without Lustre, only the base packages are resolved, and only once."""
        from_system = mocker.patch.object(
            apt.DebianPackage, "from_system", side_effect=lambda pkg: SimpleNamespace(name=pkg)
        )
        setup_repo = mocker.patch.object(ppa, "setup_lustre_repository")

        pkgs = raw_manager._packages()

        assert [pkg.name for pkg in pkgs] == list(BASE_PACKAGES)
        assert raw_manager._packages() is pkgs
        assert from_system.call_count == len(BASE_PACKAGES)
        setup_repo.assert_not_called()

    def test_lustre_repository_and_packages(
        self, raw_manager: MountsManager, mocker: MockerFixture
    ) -> None:
        """With Lustre enabled, the Lustre repository is added and its packages resolved."""
        mocker.patch.object(
            apt.DebianPackage, "from_system", side_effect=lambda pkg: SimpleNamespace(name=pkg)
        )
        setup_repo = mocker.patch.object(ppa, "setup_lustre_repository")
        raw_manager.enable_lustre = True

        pkgs = raw_manager._packages()

        setup_repo.assert_called_once_with()
        assert [pkg.name for pkg in pkgs] == list(BASE_PACKAGES + LUSTRE_PACKAGES)
        assert raw_manager._packages() is pkgs
        setup_repo.assert_called_once_with()

    def test_repository_error_wrapped(
        self, raw_manager: MountsManager, mocker: MockerFixture
    ) -> None:
        """A failure to add the Lustre repository surfaces as a manager `Error`."""
        mocker.patch.object(ppa, "setup_lustre_repository", side_effect=RepositoryError("boom"))
        raw_manager.enable_lustre = True

        with pytest.raises(Error, match="boom"):
            raw_manager._packages()

    def test_cache_invalidated_when_lustre_enabled(
        self, raw_manager: MountsManager, mocker: MockerFixture
    ) -> None:
        """Enabling Lustre after the package list was cached re-resolves it.

        Failure mode: a config change enabling Lustre would never install the
        Lustre client packages because the base-only list was cached.
        """
        mocker.patch.object(
            apt.DebianPackage, "from_system", side_effect=lambda pkg: SimpleNamespace(name=pkg)
        )
        setup_repo = mocker.patch.object(ppa, "setup_lustre_repository")
        raw_manager._packages()

        raw_manager.enable_lustre = True
        pkgs = raw_manager._packages()

        setup_repo.assert_called_once_with()
        assert any(pkg.name in LUSTRE_PACKAGES for pkg in pkgs)


class TestIsSetup:
    """Setup detection in MountsManager.is_setup()."""

    @pytest.mark.parametrize(
        "pkg_present,master_exists,autofs_exists,enable_lustre,lnet_conf_exists,expected",
        (
            pytest.param(False, True, True, False, False, False, id="missing package"),
            pytest.param(True, False, True, False, False, False, id="missing master file"),
            pytest.param(True, True, False, False, False, False, id="missing autofs file"),
            pytest.param(True, True, True, False, False, True, id="all present"),
            pytest.param(True, True, True, True, False, False, id="lustre without lnet config"),
            pytest.param(True, True, True, True, True, True, id="lustre with lnet config"),
        ),
    )
    def test_is_setup(
        self,
        manager: MountsManager,
        mocker: MockerFixture,
        tmp_path: pathlib.Path,
        pkg_present: bool,
        master_exists: bool,
        autofs_exists: bool,
        enable_lustre: bool,
        lnet_conf_exists: bool,
        expected: bool,
    ) -> None:
        """The system counts as set up only if packages, autofs files, and LNet config exist.

        Failure mode: a wrong `True` skips package installation on a fresh unit;
        a wrong `False` re-runs setup (including `lnet.init`) on every hook.
        """
        manager._packages.return_value = [SimpleNamespace(present=pkg_present)]
        manager.enable_lustre = enable_lustre
        lnet_conf = tmp_path / "lnet.conf"
        mocker.patch("utils.manager.LUSTRE_LNET_CONF", lnet_conf)
        if master_exists:
            manager._master_file.touch()
        if autofs_exists:
            manager._autofs_file.touch()
        if lnet_conf_exists:
            lnet_conf.touch()

        assert manager.is_setup() is expected


class TestSetupErrors:
    """Error wrapping in MountsManager.setup()."""

    def test_package_error_wrapped(self, manager: MountsManager) -> None:
        """An apt failure during setup surfaces as a manager `Error` with the apt message."""
        pkg = MagicMock()
        pkg.ensure.side_effect = apt.PackageError("boom")
        manager._packages.return_value = [pkg]

        with pytest.raises(Error, match="boom"):
            manager.setup()

    def test_autofs_file_error_wrapped(self, manager: MountsManager) -> None:
        """A filesystem failure while creating the autofs files surfaces as a manager `Error`."""
        manager._master_file = manager._master_file.parent / "missing" / "master"

        with pytest.raises(Error):
            manager.setup()


class TestSetupLnet:
    """LNet configuration wiring in MountsManager.setup()."""

    def test_lustre_disabled_skips_lnet(
        self, manager: MountsManager, mocker: MockerFixture
    ) -> None:
        """LNet is not configured when enable_lustre is False."""
        mock_init = mocker.patch.object(lnet, "init")
        manager.enable_lustre = False

        manager.setup()

        mock_init.assert_not_called()

    def test_networks_forwarded(self, manager: MountsManager, mocker: MockerFixture) -> None:
        """Configured LNet networks are forwarded to lnet.init."""
        mock_init = mocker.patch.object(lnet, "init")
        manager.enable_lustre = True
        manager.lnet_networks = {"tcp": ["eth0"], "o2ib": ["ib0", "ib1"]}

        manager.setup()

        mock_init.assert_called_once_with(networks=manager.lnet_networks)

    def test_lnet_error_wrapped(self, manager: MountsManager, mocker: MockerFixture) -> None:
        """An LNetError from init is wrapped as a manager Error."""
        lnet_error_message = "test lnet error message"
        mocker.patch.object(lnet, "init", side_effect=LNetError(lnet_error_message))
        manager.enable_lustre = True
        manager.lnet_networks = None

        with pytest.raises(Error, match=lnet_error_message):
            manager.setup()


class TestNfsMountOptions:
    """NFS mount options generated by Mounts.add()."""

    def test_nfs_no_version(self) -> None:
        """No `vers` option is generated when the NFS version is not set."""
        mounts = Mounts(enable_lustre=False)
        mounts.add(
            info=NfsInfo(hostname="192.168.1.1", port=None, path="/data", version=None),
            mountpoint="/nfs",
        )

        mount = mounts._mounts["/nfs"]
        assert mount.endpoint == "192.168.1.1:/data"
        assert mount.options == ["fstype=nfs"]

    def test_nfs_version(self) -> None:
        """A `vers` option is generated when the NFS version is set."""
        mounts = Mounts(enable_lustre=False)
        mounts.add(
            info=NfsInfo(hostname="192.168.1.1", port=2049, path="/data", version="4"),
            mountpoint="/nfs",
        )

        mount = mounts._mounts["/nfs"]
        assert mount.endpoint == "192.168.1.1:/data"
        assert mount.options == ["fstype=nfs", "port=2049", "vers=4"]

    def test_nfs_ipv6_hostname_bracketed(self) -> None:
        """An IPv6 NFS hostname is bracketed so autofs can parse the endpoint."""
        mounts = Mounts(enable_lustre=False)
        mounts.add(
            info=NfsInfo(hostname="2001:db8::1", port=None, path="/data", version=None),
            mountpoint="/nfs",
        )

        assert mounts._mounts["/nfs"].endpoint == "[2001:db8::1]:/data"

    def test_user_options_merged_and_sorted(self) -> None:
        """Operator mount options are merged with the generated ones and sorted."""
        mounts = Mounts(enable_lustre=False)
        mounts.add(
            info=NfsInfo(hostname="192.168.1.1", port=2049, path="/data", version="4"),
            mountpoint="/nfs",
            options=["ro", "noexec"],
        )

        assert mounts._mounts["/nfs"].options == [
            "fstype=nfs",
            "noexec",
            "port=2049",
            "ro",
            "vers=4",
        ]


class TestCephfsMountOptions:
    """CephFS mount options generated by Mounts.add()."""

    def test_cephfs_endpoint_and_options(self) -> None:
        """The CephFS endpoint embeds user, fsid, and name; monitors are `/`-joined."""
        mounts = Mounts(enable_lustre=False)
        mounts.add(
            info=CephfsInfo(
                fsid="123456789-0abc-defg-hijk-lmnopqrstuvw",
                name="cephfs",
                path="/",
                monitor_hosts=["192.168.1.1:6789", "192.168.1.2:6789"],
                user="fs-client",
                key="R//appdqz4NP4Bxcc5XWrg==",
            ),
            mountpoint="/cephfs",
        )

        mount = mounts._mounts["/cephfs"]
        assert mount.endpoint == "fs-client@123456789-0abc-defg-hijk-lmnopqrstuvw.cephfs=/"
        assert mount.options == [
            "fstype=ceph",
            "mon_addr=192.168.1.1:6789/192.168.1.2:6789",
            "secret=R//appdqz4NP4Bxcc5XWrg==",
        ]


class TestLustreMountOptions:
    """Lustre mount options generated by Mounts.add()."""

    def test_lustre_enabled(self) -> None:
        """MGS NIDs are `:`-joined into the Lustre endpoint when Lustre is enabled."""
        mounts = Mounts(enable_lustre=True)
        mounts.add(
            info=LustreInfo(mgs_ids=["10.0.0.1@tcp", "10.0.0.2@tcp"], fs_name="lustre"),
            mountpoint="/lustre",
        )

        mount = mounts._mounts["/lustre"]
        assert mount.endpoint == "10.0.0.1@tcp:10.0.0.2@tcp:/lustre"
        assert mount.options == ["fstype=lustre"]

    def test_lustre_disabled_raises(self) -> None:
        """Mounting Lustre without `enable-lustre` fails with an actionable error.

        Failure mode: without this guard the operator gets a cryptic autofs
        failure instead of being told to set `enable-lustre`.
        """
        mounts = Mounts(enable_lustre=False)

        with pytest.raises(Error, match="enable-lustre"):
            mounts.add(
                info=LustreInfo(mgs_ids=["10.0.0.1@tcp"], fs_name="lustre"),
                mountpoint="/lustre",
            )


@dataclass(frozen=True)
class _UnsupportedInfo(FilesystemInfo):
    """`FilesystemInfo` implementation with an unsupported filesystem type."""

    @classmethod
    def from_uri(cls, uri, model):
        raise NotImplementedError

    def to_uri(self, model):
        raise NotImplementedError

    @classmethod
    def filesystem_type(cls):
        return "bogus"


class TestUnsupportedFilesystem:
    """Unsupported filesystem types rejected by Mounts.add()."""

    def test_unsupported_type_raises(self) -> None:
        """An unknown filesystem type fails with an actionable error."""
        mounts = Mounts(enable_lustre=False)

        with pytest.raises(Error, match="unsupported filesystem type `bogus`"):
            mounts.add(info=_UnsupportedInfo(), mountpoint="/bogus")


class TestMounts:
    """Autofs map management in MountsManager.mounts()."""

    def test_writes_autofs_file_and_reloads(
        self, manager: MountsManager, mocker: MockerFixture, tmp_path: pathlib.Path
    ) -> None:
        """A new mount is rendered into the autofs map and autofs is reloaded."""
        systemctl = mocker.patch("utils.manager.systemctl")
        manager._autofs_file.write_text("")
        mountpoint = tmp_path / "mnt" / "nfs"

        with manager.mounts() as mounts:
            mounts.add(
                info=NfsInfo(hostname="192.168.1.1", port=None, path="/data", version=None),
                mountpoint=mountpoint,
                options=["ro"],
            )

        assert manager._autofs_file.read_text() == f"{mountpoint} -fstype=nfs,ro 192.168.1.1:/data"
        assert mountpoint.is_dir()
        systemctl.assert_called_once_with("reload-or-restart", "autofs", check=True)

    def test_unchanged_autofs_file_skips_reload(
        self, manager: MountsManager, mocker: MockerFixture, tmp_path: pathlib.Path
    ) -> None:
        """Autofs is not reloaded when the rendered map is unchanged.

        Failure mode: `relation-changed` fires often in production; reloading
        autofs on every hook would briefly disrupt existing mounts each time.
        """
        systemctl = mocker.patch("utils.manager.systemctl")
        mountpoint = tmp_path / "mnt" / "nfs"
        manager._autofs_file.write_text(f"{mountpoint} -fstype=nfs 192.168.1.1:/data")

        with manager.mounts() as mounts:
            mounts.add(
                info=NfsInfo(hostname="192.168.1.1", port=None, path="/data", version=None),
                mountpoint=mountpoint,
            )

        systemctl.assert_not_called()

    def test_changed_autofs_file_reloads(
        self, manager: MountsManager, mocker: MockerFixture, tmp_path: pathlib.Path
    ) -> None:
        """Autofs is reloaded when the rendered map changes."""
        systemctl = mocker.patch("utils.manager.systemctl")
        mountpoint = tmp_path / "mnt" / "nfs"
        manager._autofs_file.write_text(f"{mountpoint} -fstype=nfs 192.168.1.2:/data")

        with manager.mounts() as mounts:
            mounts.add(
                info=NfsInfo(hostname="192.168.1.1", port=None, path="/data", version=None),
                mountpoint=mountpoint,
            )

        systemctl.assert_called_once_with("reload-or-restart", "autofs", check=True)

    def test_exception_in_body_skips_write(
        self, manager: MountsManager, mocker: MockerFixture, tmp_path: pathlib.Path
    ) -> None:
        """An error inside the mounts block leaves the autofs map untouched.

        Pins the documented design decision: the map is not written when the
        charm enters an error state mid-handler.
        """
        mocker.patch("utils.manager.systemctl")
        manager._autofs_file.write_text("original")

        with pytest.raises(RuntimeError), manager.mounts() as mounts:
            mounts.add(
                info=NfsInfo(hostname="192.168.1.1", port=None, path="/data", version=None),
                mountpoint=tmp_path / "mnt" / "nfs",
            )
            raise RuntimeError("handler failed")

        assert manager._autofs_file.read_text() == "original"

    def test_systemd_error_wrapped(
        self, manager: MountsManager, mocker: MockerFixture, tmp_path: pathlib.Path
    ) -> None:
        """A failed autofs reload surfaces as a manager `Error` after the new map is on disk."""
        mocker.patch("utils.manager.systemctl", side_effect=SystemdError("boom"))
        manager._autofs_file.write_text("")
        mountpoint = tmp_path / "mnt" / "nfs"

        with pytest.raises(Error, match="boom"), manager.mounts() as mounts:
            mounts.add(
                info=NfsInfo(hostname="192.168.1.1", port=None, path="/data", version=None),
                mountpoint=mountpoint,
            )

        assert manager._autofs_file.read_text() == f"{mountpoint} -fstype=nfs 192.168.1.1:/data"
