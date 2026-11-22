import type { ReactNode } from "react";

function Doc({ title, updated, children }: { title: string; updated: string; children: ReactNode }) {
  return (
    <article className="mx-auto max-w-[68ch] px-5 py-12">
      <h1 className="text-[34px] font-extrabold leading-tight tracking-tight">{title}</h1>
      <p className="mt-1 text-[14px] text-muted">Last updated {updated}</p>
      <div className="mt-8 space-y-5 font-serif text-[17px] leading-[1.7] text-ink [&_h2]:mt-9 [&_h2]:font-sans [&_h2]:text-[18px] [&_h2]:font-bold">
        {children}
      </div>
    </article>
  );
}

export function Terms() {
  return (
    <Doc title="Terms of use" updated="21 November 2026">
      <p>
        CrashLens turns a dashcam clip into a 3D reconstruction, a list of measured facts and a fault split based on
        rules drawn from the Texas Transportation Code. By uploading a clip you agree to these terms.
      </p>
      <h2>What the result is</h2>
      <p>
        The result is an evidence summary. It is not legal advice, an insurance decision or an official determination
        of fault, and using it does not make anyone your lawyer. Measurements carry uncertainty, which the app shows as
        confidence values and ranges. Before relying on the result in a claim, check the facts against the video and,
        where money or liability is at stake, talk to a licensed attorney.
      </p>
      <h2>Your uploads</h2>
      <p>
        Only upload footage you recorded or have the right to use. You remain the owner of your video. You give
        CrashLens permission to process it to produce your case, including sending frames and extracted facts to the
        AI services named on the Privacy page.
      </p>
      <h2>Disputes and the record</h2>
      <p>
        Disputing a fact adds an entry to the case record; nothing is overwritten. Supporting files you attach are
        fingerprinted with SHA-256 so their integrity can be checked later. Knowingly submitting altered or false
        evidence is prohibited and may be unlawful.
      </p>
      <h2>Rule weights</h2>
      <p>
        The weights that turn rules into percentages are reasoned estimates. The calibration feature shown in the app
        uses simulated outcomes to demonstrate the method; it has not yet been fitted to real adjuster, arbitration or
        court decisions.
      </p>
      <h2>Liability</h2>
      <p>
        CrashLens is provided as is, without warranties. To the extent the law allows, its makers are not liable for
        decisions made using its output.
      </p>
    </Doc>
  );
}

export function Privacy() {
  return (
    <Doc title="Privacy" updated="21 November 2026">
      <p>This page explains what CrashLens does with the clip and details you provide.</p>
      <h2>What is stored</h2>
      <p>
        Your video, the answers on the upload form, the frames extracted around the impact, the reconstruction, the
        facts, any disputes and the files attached to them. Each case is stored on the computer running the CrashLens
        server, in its own case folder, together with the SHA-256 fingerprint of the original video and the time it
        arrived.
      </p>
      <h2>What leaves this computer</h2>
      <p>
        Two optional AI features call NVIDIA's hosted Nemotron models. The plain-English summary sends the extracted
        facts and the rules that applied, not the video. The visual cross-check sends four frames from around the
        impact. Neither feature decides the fault split. If no API key is configured, neither runs and nothing is sent.
      </p>
      <h2>What CrashLens doesn't do</h2>
      <p>
        There are no accounts, no advertising and no analytics. Case data is not sold or shared, and CrashLens does not
        contact insurers or the other driver on your behalf.
      </p>
      <h2>Deleting a case</h2>
      <p>
        Delete the case's folder under <span className="font-sans text-[15px]">backend/data/cases</span> on the
        server. Copies you exported, such as a printed report, are yours to keep or destroy.
      </p>
    </Doc>
  );
}
