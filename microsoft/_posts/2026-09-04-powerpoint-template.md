---
title: "How to Apply a PowerPoint Template to an Existing Presentation"
categories: [microsoft]
tags: [powerpoint, office, templates]
---

There are two different jobs that are often described as “applying a template”:

- Apply a template to a new presentation, then add your content.
- Move existing slides into a presentation that already uses the template.

For an existing deck, the second job is more reliable when done as an import. Microsoft recommends starting with a fresh presentation created from the template, then importing the existing slides into it.

## Apply the template to a new presentation

1. Select **File > New**.
2. Open the **Personal**, **Custom**, or equivalent template tab. The name depends on your PowerPoint version.
3. Select the `.potx` template and choose **Create**.
4. Add new slides to this presentation and move your content into the template's layouts.

This gives the new presentation the template's theme, slide master, and layouts from the beginning. A `.potx` file is a PowerPoint template; it can contain both a theme and placeholder instructions for future slides.

## Move existing slides into the template

If your content is already in another `.pptx` file, keep the template presentation as the destination:

1. Open the presentation that contains your existing slides.
2. In the thumbnail pane, select the slides to move. Use `Ctrl+A` for all slides, or hold `Ctrl` while selecting individual slides.
3. Copy the slides with `Ctrl+C`.
4. Switch to the new presentation created from the template.
5. Right-click in the destination thumbnail pane and choose **Use Destination Theme** under **Paste Options**.

**Use Destination Theme** makes the imported slides follow the template presentation's design. This is usually the right choice when the goal is to restyle existing content with a new template.

## When to keep the original design

If a slide must retain the colors, fonts, backgrounds, and other formatting from its original presentation, choose **Keep Source Formatting** instead. The slide will remain visually tied to the source design, so it may not look consistent with the rest of the new deck.

PowerPoint also provides **Home > New Slide > Reuse Slides**. In the Reuse Slides pane, choose the source presentation and select **Keep source formatting** when the original appearance must be preserved. When that option is not selected, imported slides inherit the styling of the destination presentation.

| Goal | Paste or import option |
| --- | --- |
| Make old slides fit the new template | **Use Destination Theme** |
| Preserve the source deck's appearance | **Keep Source Formatting** |

## Save the design for later

If this design will be reused, save the source presentation as a template:

1. Select **File > Save As** (or **Save a Copy** in some Microsoft 365 versions).
2. Choose **PowerPoint Template** in the file-type list.
3. Save the file with the `.potx` extension.

You can then find it under **File > New**, usually in **Personal**, **Custom**, or **Custom Office Templates**, depending on the PowerPoint version and configuration.

## References

- [Apply a template to an existing presentation](https://support.microsoft.com/en-us/powerpoint/apply-a-template-to-an-existing-presentation) — Microsoft’s recommended workflow for starting with a fresh template and importing existing slides, including the **Use Destination Theme** option.
- [Reuse (import) slides from another presentation](https://support.microsoft.com/en-us/powerpoint/reuse-import-slides-from-another-presentation) — Documents **Reuse Slides**, local presentation import, and **Keep source formatting**.
- [Create and save a PowerPoint template](https://support.microsoft.com/en-us/powerpoint/create-and-save-a-powerpoint-template) — Documents creating a template and saving it as a `.potx` file.
- [Save a slide design (theme) as a template in PowerPoint](https://support.microsoft.com/en-us/powerpoint/save-a-slide-design-theme-as-a-template-in-powerpoint) — Explains the difference between a theme and a template and how to reuse the result.
