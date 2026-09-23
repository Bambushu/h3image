"""CPU-only regression tests: no ComfyUI, no GPU. Each test pins a bug found in the 2026-09-14 review;
renders are mocked by a fake run() that writes a solid image."""
import json, os, subprocess, sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import h3edit  # noqa: E402
import h3_inpaint  # noqa: E402


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(h3edit, "INPUT_DIR", str(tmp_path / "comfy_input"))      # deliberately absent
    monkeypatch.setattr(h3edit, "OUTPUT_DIR", str(tmp_path / "comfy_output"))
    monkeypatch.setattr(h3edit, "_WORK", None)
    monkeypatch.setenv("H3EDIT_TRANSPORT", "shared")
    h3edit.apply_profile("mac")
    yield
    h3edit.apply_profile("mac")


def img(path, size=(512, 512), color=(120, 120, 120)):
    Image.new("RGB", size, color).save(path)
    return str(path)


def fake_run(calls, color=(200, 30, 30)):
    def run(a):
        calls.append({"refs": list(a.refs), "seed": a.seed, "steps": a.steps})
        out = os.path.join(h3edit.workdir(), f"render_{len(calls)}.png")
        return img(out, (1024, 1024), color)
    return run


def main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["h3edit", *argv])
    return h3edit.main()


def test_graphs_ship_as_package_data():
    assert os.path.exists(h3edit.GRAPH) and os.path.exists(h3edit.CUDA_GRAPH)
    assert os.path.dirname(h3edit.GRAPH).endswith("h3edit_graphs")


def test_same_basename_refs_do_not_collide(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    ra = img(tmp_path / "a" / "ref.png", color=(255, 0, 0))
    rb = img(tmp_path / "b" / "ref.png", color=(0, 0, 255))
    na, nb = h3edit.stage_ref(ra), h3edit.stage_ref(rb)
    assert na != nb
    assert Image.open(os.path.join(h3edit.INPUT_DIR, na)).getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(os.path.join(h3edit.INPUT_DIR, nb)).getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.parametrize("mode", ["--detail", "--inpaint"])
def test_two_seed_batch_completes_and_refs_do_not_accumulate(tmp_path, monkeypatch, mode):
    calls = []
    monkeypatch.setattr(h3edit, "run", fake_run(calls))
    src = img(tmp_path / "src.png")
    art = img(tmp_path / "art.png")
    out = str(tmp_path / "out.png")
    main(monkeypatch, "edit", mode, "128,128,384,384", "--source", src, "-r", art, "-o", out,
         "--seeds", "7,8")
    assert len(calls) == 2
    assert len(calls[0]["refs"]) == len(calls[1]["refs"])
    assert os.path.exists(str(tmp_path / "out_s7.png")) and os.path.exists(str(tmp_path / "out_s8.png"))
    assert os.path.exists(str(tmp_path / "out_sheet.png"))


def test_tone_match_uniform_ring_is_not_black():
    src = Image.new("RGB", (512, 512), (90, 90, 90))
    ren = Image.new("RGB", (512, 512), (150, 150, 150))
    with np.errstate(all="raise"):
        out = h3edit.tone_match(ren, src, (192, 192, 320, 320), 32)
    assert out.getpixel((256, 256)) != (0, 0, 0)


def test_small_detail_box_still_changes_pixels(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(h3edit, "run", fake_run(calls))
    src = img(tmp_path / "src.png")
    out = str(tmp_path / "out.png")
    main(monkeypatch, "fix", "--detail", "64,64,128,128", "--source", src, "-o", out, "--seed", "1")
    assert len(calls) == 1
    assert Image.open(out).getpixel((96, 96)) != Image.open(src).getpixel((96, 96))


def test_successful_render_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(h3edit, "run", lambda a: img(tmp_path / "o.png"))
    assert main(monkeypatch, "edit", "-r", img(tmp_path / "r.png"), "--seed", "1", "-o", str(tmp_path / "o.png")) is None


def test_single_seeds_value_is_used(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(h3edit, "dispatch", lambda a: seen.append(a.seed))
    main(monkeypatch, "edit", "-r", img(tmp_path / "r.png"), "--seed", "42", "--seeds", "123")
    assert seen == [123]


def test_explicit_cuda_steps_are_honored(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(h3edit, "dispatch", lambda a: built.append(h3edit.build(a)))
    main(monkeypatch, "--profile", "cuda", "edit", "-r", img(tmp_path / "r.png"), "--steps", "8", "--seed", "1")
    g = built[0]
    assert g["7"]["inputs"]["steps"] == 8
    built.clear()
    main(monkeypatch, "--profile", "cuda", "edit", "-r", img(tmp_path / "r.png"), "--seed", "1")
    assert built[0]["7"]["inputs"]["steps"] == 20 and built[0]["6"]["inputs"]["sampler_name"] == "euler"


def test_dry_run_refuses_modes_that_would_render(tmp_path, monkeypatch):
    posted = []
    monkeypatch.setattr(h3edit, "api", lambda path, payload=None: posted.append(path) or {})
    with pytest.raises(SystemExit) as e:
        main(monkeypatch, "edit", "-r", img(tmp_path / "r.png"), "--dry-run")
    assert e.value.code == 2 and posted == []


def test_detail_and_inpaint_are_exclusive(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        main(monkeypatch, "x", "--detail", "0,0,64,64", "--inpaint", "0,0,64,64",
             "--source", img(tmp_path / "s.png"), "-r", img(tmp_path / "r.png"), "-o", str(tmp_path / "o.png"))


def test_generate_without_local_comfy_input_dir(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(h3edit, "dispatch", lambda a: calls.append(list(a.refs)))
    main(monkeypatch, "a red sailboat", "--generate", "--seed", "1")
    assert calls and os.path.exists(calls[0][0])


def test_outpaint_plan_matches_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(h3edit, "inpaint_run", lambda a: None)       # margins stay edge-replicated
    src = img(tmp_path / "src.png", (512, 512))
    out = str(tmp_path / "out.png")
    main(monkeypatch, "a beach", "--outpaint", "0,0,33,0", "--source", src, "-o", out, "--seed", "1")
    planned = [l for l in capsys.readouterr().out.splitlines() if l.startswith("outpaint:")][0]
    w, h = Image.open(out).size
    assert f"-> {w}x{h}" in planned


def test_canvas_project_opens_from_another_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img(tmp_path / "start.png", (640, 384))
    monkeypatch.setattr(sys, "argv", ["h3-inpaint", "init", "proj", "--canvas", "start.png"])
    h3_inpaint.main()
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    p, plan, state = h3_inpaint.load(str(tmp_path / "proj"))
    assert os.path.exists(plan["canvas"]) and os.path.exists(state["canvas"])
    assert not os.path.isabs(json.load(open(tmp_path / "proj" / "plan.json"))["canvas"])


def test_reinit_refuses_to_wipe_a_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img(tmp_path / "start.png", (640, 384))
    monkeypatch.setattr(sys, "argv", ["h3-inpaint", "init", "proj", "--canvas", "start.png"]); h3_inpaint.main()
    monkeypatch.setattr(sys, "argv", ["h3-inpaint", "add", "proj", "p1", "--box", "0,0,64,64", "--prompt", "x"]); h3_inpaint.main()
    monkeypatch.setattr(sys, "argv", ["h3-inpaint", "init", "proj", "--canvas", "start.png"])
    with pytest.raises(SystemExit):
        h3_inpaint.main()
    assert [e["name"] for e in json.load(open(tmp_path / "proj" / "plan.json"))["passes"]] == ["p1"]


def test_cuda_doctor_fails_without_models(monkeypatch):
    h3edit.apply_profile("cuda")
    monkeypatch.setattr(h3edit, "api", lambda path, payload=None: {} if "object_info" not in path
                        else {path.rsplit("/", 1)[1]: {}})
    monkeypatch.setattr(h3edit, "combo_options", lambda node, widget: [])
    assert h3edit.doctor_cuda() is False


def test_pod_refuses_detail_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img(tmp_path / "start.png", (640, 384))
    monkeypatch.setattr(sys, "argv", ["h3-inpaint", "init", "proj", "--canvas", "start.png"]); h3_inpaint.main()
    p, plan, state = h3_inpaint.load("proj")
    e = {"name": "d", "box": [0, 0, 64, 64], "prompt": "x", "kind": "detail"}
    with pytest.raises(SystemExit, match="local-only"):
        h3_inpaint.run_pass(p, plan, state, e, pod_url="http://pod")


@pytest.mark.parametrize("wf", sorted(os.listdir(os.path.join(ROOT, "workflows"))))
def test_gui_workflows_are_complete(wf):
    w = json.load(open(os.path.join(ROOT, "workflows", wf)))
    nodes = {n["id"]: n for n in w["nodes"]}
    assert [g["title"][0] for g in w["groups"]] == ["1", "2", "3", "4"]
    assert any(n["type"] == "MarkdownNote" for n in nodes.values())
    length = [n for n in nodes.values() if n["type"] == "PrimitiveInt" and n.get("title", "").startswith("length")]
    assert length and length[0]["widgets_values"][0] == 1
    r2v = next(n for n in nodes.values() if n["type"] == "MiniMaxH3ReferenceToVideo")
    assert any(i["name"] == "ref_images.ref_image_0" and i.get("link") is not None for i in r2v["inputs"])
    if "inpaint" in wf:
        assert {"H3V2VInit", "ImageCompositeMasked", "GetImageSize"} <= {n["type"] for n in nodes.values()}
